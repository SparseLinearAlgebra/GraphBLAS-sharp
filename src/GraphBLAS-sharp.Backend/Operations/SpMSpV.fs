namespace GraphBLAS.FSharp.Backend.Operations

open Brahma.FSharp
open GraphBLAS.FSharp.Backend.Common
open GraphBLAS.FSharp.Backend.Quotes
open Microsoft.FSharp.Quotations
open GraphBLAS.FSharp.Objects
open GraphBLAS.FSharp.Objects.ClVector
open GraphBLAS.FSharp.Objects.ClContextExtensions
open GraphBLAS.FSharp.Objects.ArraysExtensions
open GraphBLAS.FSharp.Objects.ClCellExtensions

module SpMSpV =

    //For v in vectorIndices collect R[v] and R[v + 1]
    let private collectRows (clContext: ClContext) workGroupSize =

        let collectRows =
            <@ fun (ndRange: Range1D) inputSize (vectorIndices: ClArray<int>) (rowOffsets: ClArray<int>) (resultArray: ClArray<int>) ->

                let i = ndRange.GlobalID0

                //resultArray is twice vector size
                if i < (inputSize * 2) then
                    let columnIndex = vectorIndices.[i / 2]

                    if i % 2 = 0 then
                        resultArray.[i] <- rowOffsets.[columnIndex]
                    else
                        resultArray.[i] <- rowOffsets.[columnIndex + 1]
                elif i = inputSize * 2 then
                    resultArray.[i] <- 0 @>

        let collectRows = clContext.Compile collectRows

        fun (queue: RawCommandQueue) size (vectorIndices: ClArray<int>) (rowOffsets: ClArray<int>) ->

            let ndRange =
                Range1D.CreateValid(size * 2 + 1, workGroupSize)

            // Last element will contain length of array for gather
            let resultRows =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, size * 2 + 1)

            let collectRows = collectRows.GetKernel()

            collectRows.KernelFunc ndRange size vectorIndices rowOffsets resultRows

            queue.RunKernel(collectRows)

            resultRows

    //For above array compute result offsets
    let private computeOffsetsInplace (clContext: ClContext) workGroupSize =

        let prepareOffsets =
            <@ fun (ndRange: Range1D) inputSize (inputArray: ClArray<int>) ->

                let i = ndRange.GlobalID0

                if i < inputSize && i % 2 = 0 then
                    inputArray.[i + 1] <- inputArray.[i + 1] - inputArray.[i]
                    inputArray.[i] <- 0 @>

        let sum =
            ScanInternal.standardExcludeInPlace clContext workGroupSize

        let prepareOffsets = clContext.Compile prepareOffsets

        fun (queue: RawCommandQueue) size (input: ClArray<int>) ->

            let ndRange = Range1D.CreateValid(size, workGroupSize)

            let prepareOffsets = prepareOffsets.GetKernel()

            prepareOffsets.KernelFunc ndRange size input

            queue.RunKernel(prepareOffsets)

            let resSize = (sum queue input).ToHostAndFree queue

            resSize

    //Gather rows from the matrix that will be used in multiplication
    let private gather (clContext: ClContext) workGroupSize =

        let gather =
            <@ fun (ndRange: Range1D) vectorNNZ (rowOffsets: ClArray<int>) (matrixRowPointers: ClArray<int>) (matrixColumns: ClArray<int>) (matrixValues: ClArray<'a>) (vectorIndices: ClArray<int>) (resultRowsArray: ClArray<int>) (resultIndicesArray: ClArray<int>) (resultValuesArray: ClArray<'a>) ->

                //Serial number of row to gather
                let row = ndRange.GlobalID0

                if row < vectorNNZ then
                    let offsetIndex = row * 2 + 1
                    let rowOffset = rowOffsets.[offsetIndex]

                    //vectorIndices.[row] --- actual number of row in matrix
                    let actualRow = vectorIndices.[row]
                    let matrixIndexOffset = matrixRowPointers.[actualRow]

                    if rowOffset <> rowOffsets.[offsetIndex + 1] then
                        let rowSize = rowOffsets.[offsetIndex + 1] - rowOffset

                        for i in 0 .. rowSize - 1 do
                            resultRowsArray.[i + rowOffset] <- actualRow
                            resultIndicesArray.[i + rowOffset] <- matrixColumns.[matrixIndexOffset + i]
                            resultValuesArray.[i + rowOffset] <- matrixValues.[matrixIndexOffset + i] @>

        let collectRows = collectRows clContext workGroupSize

        let computeOffsetsInplace =
            computeOffsetsInplace clContext workGroupSize

        let gather = clContext.Compile gather

        fun (queue: RawCommandQueue) (matrix: ClMatrix.CSR<'a>) (vector: ClVector.Sparse<'b>) ->

            //Collect R[v] and R[v + 1] for each v in vector
            let collectedRows =
                collectRows queue vector.NNZ vector.Indices matrix.RowPointers

            //Place R[v + 1] - R[v] in previous array and do prefix sum, computing offsets for gather array
            let gatherArraySize =
                computeOffsetsInplace queue (vector.NNZ * 2 + 1) collectedRows

            if gatherArraySize = 0 then
                collectedRows.Free()
                None
            else
                let ndRange =
                    Range1D.CreateValid(vector.NNZ, workGroupSize)

                let gather = gather.GetKernel()

                let resultRows =
                    clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, gatherArraySize)

                let resultIndices =
                    clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, gatherArraySize)

                let resultValues =
                    clContext.CreateClArrayWithSpecificAllocationMode<'a>(DeviceOnly, gatherArraySize)

                if gatherArraySize > 0 then
                    gather.KernelFunc
                        ndRange
                        vector.NNZ
                        collectedRows
                        matrix.RowPointers
                        matrix.Columns
                        matrix.Values
                        vector.Indices
                        resultRows
                        resultIndices
                        resultValues

                    queue.RunKernel gather

                collectedRows.Free()

                Some(resultRows, resultIndices, resultValues)


    let private multiplyScalar (clContext: ClContext) (mul: Expr<'a option -> 'b option -> 'c option>) workGroupSize =

        let multiply =
            <@ fun (ndRange: Range1D) resultLength vectorLength (rowIndices: ClArray<int>) (matrixValues: ClArray<'a>) (vectorIndices: ClArray<int>) (vectorValues: ClArray<'b>) (resultValues: ClArray<'c option>) ->
                let i = ndRange.GlobalID0

                if i < resultLength then
                    let index = rowIndices.[i]
                    let matrixValue = matrixValues.[i]

                    let vectorValue =
                        (%Search.Bin.byKey) vectorLength index vectorIndices vectorValues

                    let res = (%mul) (Some matrixValue) vectorValue
                    resultValues.[i] <- res @>

        let multiply = clContext.Compile multiply

        fun (queue: RawCommandQueue) (columnIndices: ClArray<int>) (matrixValues: ClArray<'a>) (vector: Sparse<'b>) ->

            let resultLength = columnIndices.Length

            let result =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, resultLength)

            let ndRange =
                Range1D.CreateValid(resultLength, workGroupSize)

            let multiply = multiply.GetKernel()

            multiply.KernelFunc
                ndRange
                resultLength
                vector.NNZ
                columnIndices
                matrixValues
                vector.Indices
                vector.Values
                result

            queue.RunKernel(multiply)

            result

    let run
        (add: Expr<'c option -> 'c option -> 'c option>)
        (mul: Expr<'a option -> 'b option -> 'c option>)
        (clContext: ClContext)
        workGroupSize
        =

        //TODO: Common.Gather?
        let gather = gather clContext workGroupSize

        let sort =
            Sort.Bitonic.sortRowsColumnsValuesInplace clContext workGroupSize

        let multiplyScalar =
            multiplyScalar clContext mul workGroupSize

        let segReduce =
            Reduce.ByKey.Option.segmentSequential add clContext workGroupSize

        fun (queue: RawCommandQueue) (matrix: ClMatrix.CSR<'a>) (vector: ClVector.Sparse<'b>) ->
            gather queue matrix vector
            |> Option.map
                (fun (gatherRows, gatherIndices, gatherValues) ->
                    sort queue gatherIndices gatherRows gatherValues

                    let sortedRows, sortedIndices, sortedValues = gatherRows, gatherIndices, gatherValues

                    let multipliedValues =
                        multiplyScalar queue sortedRows sortedValues vector

                    sortedValues.Free()
                    sortedRows.Free()

                    let result =
                        segReduce queue DeviceOnly sortedIndices multipliedValues
                        |> Option.map
                            (fun (reducedValues, reducedKeys) ->

                                { Context = clContext
                                  Indices = reducedKeys
                                  Values = reducedValues
                                  Size = matrix.ColumnCount })

                    multipliedValues.Free()
                    sortedIndices.Free()

                    result)
            |> Option.bind id


    let runBoolStandard
        (add: Expr<'c option -> 'c option -> 'c option>)
        (mul: Expr<'a option -> 'b option -> 'c option>)
        (clContext: ClContext)
        workGroupSize
        =

        let gather = gather clContext workGroupSize

        let sort =
            Sort.Bitonic.sortRowsColumnsValuesInplace clContext workGroupSize

        let removeDuplicates =
            GraphBLAS.FSharp.ClArray.removeDuplications clContext workGroupSize

        let create =
            GraphBLAS.FSharp.ClArray.create clContext workGroupSize

        fun (queue: RawCommandQueue) (matrix: ClMatrix.CSR<'a>) (vector: ClVector.Sparse<'b>) ->

            gather queue matrix vector
            |> Option.map
                (fun (gatherRows, gatherIndices, gatherValues) ->
                    sort queue gatherIndices gatherRows gatherValues

                    let resultIndices = removeDuplicates queue gatherIndices

                    gatherIndices.Free()
                    gatherRows.Free()
                    gatherValues.Free()

                    { Context = clContext
                      Indices = resultIndices
                      Values = create queue DeviceOnly resultIndices.Length true
                      Size = matrix.ColumnCount })

    module Masked =

        let private count (clContext: ClContext) workGroupSize =

            let count =
                <@ fun (ndRange: Range1D) vectorLength (vectorIndices: ClArray<int>) (vectorMask: ClArray<'d option>) (matrixRowPointers: ClArray<int>) (matrixColumns: ClArray<int>) (result: ClCell<int>) ->
                    let gid = ndRange.GlobalID0
                    let step = ndRange.GlobalWorkSize

                    let mutable idx = gid

                    while idx < vectorLength do
                        let vectorIndex = vectorIndices.[idx]

                        let rowStart = matrixRowPointers.[vectorIndex]
                        let rowEnd = matrixRowPointers.[vectorIndex + 1]

                        let mutable count = 0

                        for i in rowStart .. rowEnd - 1 do
                            match vectorMask.[matrixColumns.[i]] with
                            | None -> count <- count + 1
                            | Some _ -> ()

                        atomic (+) result.Value count |> ignore

                        idx <- idx + step @>

            let count = clContext.Compile count

            fun (queue: RawCommandQueue) (matrix: ClMatrix.CSR<'a>) (vector: ClVector.Sparse<'b>) (vectorMask: ClArray<'d option>) ->

                let length = vector.NNZ

                let numberOfGroups =
                    Utils.divUpClamp length workGroupSize 1 1024

                let result = clContext.CreateClCell(0)

                let ndRange =
                    Range1D.CreateValid(numberOfGroups * workGroupSize, workGroupSize)

                let count = count.GetKernel()

                count.KernelFunc ndRange length vector.Indices vectorMask matrix.RowPointers matrix.Columns result

                queue.RunKernel count

                result.ToHostAndFree queue

        let private multiplyValues
            (clContext: ClContext)
            (mul: Expr<'a option -> 'b option -> 'c option>)
            workGroupSize
            =

            let multiply =
                <@ fun (ndRange: Range1D) resultLength (vectorIndices: ClArray<int>) (vectorValues: ClArray<'b>) (vectorMask: ClArray<'d option>) (matrixRowPointers: ClArray<int>) (matrixColumns: ClArray<int>) (matrixValues: ClArray<'a>) (resultOffset: ClCell<int>) (resultIndices: ClArray<int>) (resultValues: ClArray<'c option>) ->
                    let gid = ndRange.GlobalID0
                    let step = ndRange.GlobalWorkSize

                    let mutable i = gid

                    while i < resultLength do
                        let vectorIndex = vectorIndices.[i]
                        let vectorValue = vectorValues.[i]

                        let rowStart = matrixRowPointers.[vectorIndex]
                        let rowEnd = matrixRowPointers.[vectorIndex + 1]

                        let mutable count = 0

                        for i in rowStart .. rowEnd - 1 do
                            match vectorMask.[matrixColumns.[i]] with
                            | None -> count <- count + 1
                            | Some _ -> ()

                        let mutable offset = atomic (+) resultOffset.Value count

                        for i in rowStart .. rowEnd - 1 do
                            let columnIndex = matrixColumns.[i]

                            // TODO: Pass mask operation
                            match vectorMask.[columnIndex] with
                            | None ->
                                resultIndices.[offset] <- columnIndex
                                resultValues.[offset] <- (%mul) (Some matrixValues.[i]) (Some vectorValue)
                                offset <- offset + 1
                            | Some _ -> ()

                        i <- i + step @>

            let kernel = clContext.Compile multiply

            fun (queue: RawCommandQueue) (matrix: ClMatrix.CSR<'a>) (vector: ClVector.Sparse<'b>) (vectorMask: ClArray<'d option>) (resultSize: int) ->

                let multipliedIndices =
                    clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, resultSize)

                let multipliedValues =
                    clContext.CreateClArrayWithSpecificAllocationMode<'c option>(DeviceOnly, resultSize)

                let offset = clContext.CreateClCell 0

                let numberOfGroups =
                    Utils.divUpClamp vector.NNZ workGroupSize 1 1024

                let ndRange =
                    Range1D.CreateValid(numberOfGroups * workGroupSize, workGroupSize)

                let kernel = kernel.GetKernel()

                kernel.KernelFunc
                    ndRange
                    vector.NNZ
                    vector.Indices
                    vector.Values
                    vectorMask
                    matrix.RowPointers
                    matrix.Columns
                    matrix.Values
                    offset
                    multipliedIndices
                    multipliedValues

                queue.RunKernel kernel

                offset.Free()

                multipliedIndices, multipliedValues

        let runMasked
            (add: Expr<'c option -> 'c option -> 'c option>)
            (mul: Expr<'a option -> 'b option -> 'c option>)
            (clContext: ClContext)
            workGroupSize
            =

            let count = count clContext workGroupSize

            let multiplyValues =
                multiplyValues clContext mul workGroupSize

            let sort =
                Sort.Bitonic.sortKeyValuesInplace clContext workGroupSize

            let segReduce =
                Reduce.ByKey.Option.segmentSequential add clContext workGroupSize

            fun (queue: RawCommandQueue) (matrix: ClMatrix.CSR<'a>) (vector: ClVector.Sparse<'b>) (mask: ClArray<'d option>) ->

                match count queue matrix vector mask with
                | 0 -> None
                | resultSize ->
                    let multipliedIndices, multipliedValues =
                        multiplyValues queue matrix vector mask resultSize

                    sort queue multipliedIndices multipliedValues

                    let result =
                        segReduce queue DeviceOnly multipliedIndices multipliedValues
                        |> Option.map
                            (fun (reducedValues, reducedKeys) ->
                                { Context = clContext
                                  Indices = reducedKeys
                                  Values = reducedValues
                                  Size = matrix.ColumnCount })

                    multipliedIndices.Free()
                    multipliedValues.Free()

                    result

        let runMaskedBoolStandard
            (add: Expr<'c option -> 'c option -> 'c option>)
            (mul: Expr<'a option -> 'b option -> 'c option>)
            (clContext: ClContext)
            workGroupSize
            =

            let count = count clContext workGroupSize

            let multiplyValues =
                multiplyValues clContext mul workGroupSize

            let sort =
                Sort.Bitonic.sortKeyValuesInplace clContext workGroupSize

            let removeDuplicates =
                GraphBLAS.FSharp.ClArray.removeDuplications clContext workGroupSize

            let create =
                GraphBLAS.FSharp.ClArray.create clContext workGroupSize

            fun (queue: RawCommandQueue) (matrix: ClMatrix.CSR<'a>) (vector: ClVector.Sparse<'b>) (mask: ClArray<'d option>) ->

                match count queue matrix vector mask with
                | 0 -> None
                | resultSize ->
                    let multipliedIndices, multipliedValues =
                        multiplyValues queue matrix vector mask resultSize

                    sort queue multipliedIndices multipliedValues

                    let resultIndices = removeDuplicates queue multipliedIndices

                    multipliedIndices.Free()
                    multipliedValues.Free()

                    Some
                    <| { Context = clContext
                         Indices = resultIndices
                         Values = create queue DeviceOnly resultIndices.Length true
                         Size = matrix.ColumnCount }
