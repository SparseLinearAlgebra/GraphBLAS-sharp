namespace GraphBLAS.FSharp.Backend.Operations

open FSharpx.Collections
open Microsoft.FSharp.Quotations
open FSharp.Quotations.Evaluator.QuotationEvaluationExtensions
open Brahma.FSharp
open GraphBLAS.FSharp
open GraphBLAS.FSharp.Backend.Quotes
open GraphBLAS.FSharp.Backend.Matrix.COO
open GraphBLAS.FSharp.Backend.Matrix.CSR
open GraphBLAS.FSharp.Objects.ClMatrix
open GraphBLAS.FSharp.Objects.ClCellExtensions
open GraphBLAS.FSharp.Objects.ClContextExtensions
open GraphBLAS.FSharp.Objects.ArraysExtensions

module internal Kronecker =
    let private updateBitmap (clContext: ClContext) workGroupSize op =

        let updateBitmap (op: Expr<'a option -> 'b option -> 'c option>) =
            <@ fun (ndRange: Range1D) (operand: ClCell<'a>) valuesLength zeroCount (values: ClArray<'b>) (resultBitmap: ClArray<int>) ->

                let gid = ndRange.GlobalID0

                if gid = 0 then

                    let item = resultBitmap.[0]
                    let newItem = item + zeroCount

                    match (%op) (Some operand.Value) None with
                    | Some _ -> resultBitmap.[0] <- newItem
                    | _ -> ()

                elif (gid - 1) < valuesLength then

                    let item = resultBitmap.[gid]
                    let newItem = item + 1

                    match (%op) (Some operand.Value) (Some values.[gid - 1]) with
                    | Some _ -> resultBitmap.[gid] <- newItem
                    | _ -> () @>

        let updateBitmap = clContext.Compile <| updateBitmap op

        fun (processor: MailboxProcessor<_>) (operand: ClCell<'a>) (matrixRight: CSR<'b>) (bitmap: ClArray<int>) ->

            let resultLength = matrixRight.NNZ + 1

            let ndRange =
                Range1D.CreateValid(resultLength, workGroupSize)

            let updateBitmap = updateBitmap.GetKernel()

            let numberOfZeros =
                matrixRight.ColumnCount * matrixRight.RowCount
                - matrixRight.NNZ

            processor.Post(
                Msg.MsgSetArguments
                    (fun () ->
                        updateBitmap.KernelFunc ndRange operand matrixRight.NNZ numberOfZeros matrixRight.Values bitmap)
            )

            processor.Post(Msg.CreateRunMsg<_, _> updateBitmap)

    let private getAllocationSize (clContext: ClContext) workGroupSize op =

        let updateBitmap = updateBitmap clContext workGroupSize op

        let sum =
            Common.Reduce.sum <@ fun x y -> x + y @> 0 clContext workGroupSize

        let item = ClArray.item clContext workGroupSize

        let createClArray =
            ClArray.zeroCreate clContext workGroupSize

        let opOnHost = op.Evaluate()

        fun (queue: MailboxProcessor<_>) (matrixZero: COO<'c> option) (matrixLeft: CSR<'a>) (matrixRight: CSR<'b>) ->

            let nnz =
                match opOnHost None None with
                | Some _ ->
                    let leftZeroCount =
                        matrixLeft.RowCount * matrixLeft.ColumnCount
                        - matrixLeft.NNZ

                    let rightZeroCount =
                        matrixRight.RowCount * matrixRight.ColumnCount
                        - matrixRight.NNZ

                    leftZeroCount * rightZeroCount
                | _ -> 0

            let bitmap =
                createClArray queue DeviceOnly (matrixRight.NNZ + 1)

            for index in 0 .. matrixLeft.NNZ - 1 do
                let value = item queue index matrixLeft.Values

                updateBitmap queue value matrixRight bitmap

                value.Free queue

            let bitmapSum = sum queue bitmap

            bitmap.Free queue

            let leftZeroCount =
                matrixLeft.ColumnCount * matrixLeft.RowCount
                - matrixLeft.NNZ

            match matrixZero with
            | Some m -> m.NNZ * leftZeroCount
            | _ -> 0
            + nnz
            + bitmapSum.ToHostAndFree queue

    let private preparePositions<'a, 'b, 'c when 'b: struct> (clContext: ClContext) workGroupSize op =

        let preparePositions (op: Expr<'a option -> 'b option -> 'c option>) =
            <@ fun (ndRange: Range1D) (operand: ClCell<'a>) rowCount columnCount (values: ClArray<'b>) (rowPointers: ClArray<int>) (columns: ClArray<int>) (resultBitmap: ClArray<int>) (resultValues: ClArray<'c>) ->

                let gid = ndRange.GlobalID0

                if gid < rowCount * columnCount then

                    let columnIndex = gid % columnCount
                    let rowIndex = gid / columnCount

                    let firstIndex = rowPointers.[rowIndex]
                    let lastIndex = rowPointers.[rowIndex + 1] - 1

                    let value =
                        (%Search.Bin.inRange) firstIndex lastIndex columnIndex columns values

                    match (%op) (Some operand.Value) value with
                    | Some resultValue ->
                        resultValues.[gid] <- resultValue
                        resultBitmap.[gid] <- 1
                    | None -> resultBitmap.[gid] <- 0 @>

        let kernel = clContext.Compile <| preparePositions op

        fun (processor: MailboxProcessor<_>) (operand: ClCell<'a>) (matrix: CSR<'b>) (resultDenseMatrix: ClArray<'c>) (resultBitmap: ClArray<int>) ->

            let resultLength = matrix.RowCount * matrix.ColumnCount

            let ndRange =
                Range1D.CreateValid(resultLength, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(
                Msg.MsgSetArguments
                    (fun () ->
                        kernel.KernelFunc
                            ndRange
                            operand
                            matrix.RowCount
                            matrix.ColumnCount
                            matrix.Values
                            matrix.RowPointers
                            matrix.Columns
                            resultBitmap
                            resultDenseMatrix)
            )

            processor.Post(Msg.CreateRunMsg<_, _> kernel)

    let private setPositions<'c when 'c: struct> (clContext: ClContext) workGroupSize =

        let setPositions =
            <@ fun (ndRange: Range1D) rowCount columnCount startIndex (rowOffset: int) (columnOffset: int) (bitmap: ClArray<int>) (values: ClArray<'c>) (resultRows: ClArray<int>) (resultColumns: ClArray<int>) (resultValues: ClArray<'c>) ->

                let gid = ndRange.GlobalID0

                if gid < rowCount * columnCount
                   && (gid = 0 && bitmap.[gid] = 1
                       || gid > 0 && bitmap.[gid - 1] < bitmap.[gid]) then

                    let columnIndex = gid % columnCount
                    let rowIndex = gid / columnCount

                    let index = startIndex + bitmap.[gid] - 1

                    resultRows.[index] <- rowIndex + rowOffset
                    resultColumns.[index] <- columnIndex + columnOffset
                    resultValues.[index] <- values.[gid] @>

        let kernel = clContext.Compile <| setPositions

        let scan =
            Common.PrefixSum.standardIncludeInPlace clContext workGroupSize

        fun (processor: MailboxProcessor<_>) rowCount columnCount (rowOffset: int) (columnOffset: int) (startIndex: int) (resultMatrix: COO<'c>) (values: ClArray<'c>) (bitmap: ClArray<int>) ->

            let sum = scan processor bitmap

            let ndRange =
                Range1D.CreateValid(rowCount * columnCount, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(
                Msg.MsgSetArguments
                    (fun () ->
                        kernel.KernelFunc
                            ndRange
                            rowCount
                            columnCount
                            startIndex
                            rowOffset
                            columnOffset
                            bitmap
                            values
                            resultMatrix.Rows
                            resultMatrix.Columns
                            resultMatrix.Values)
            )

            processor.Post(Msg.CreateRunMsg<_, _> kernel)

            (sum.ToHostAndFree processor) + startIndex

    let private copyToResult (clContext: ClContext) workGroupSize =

        let copyToResult =
            <@ fun (ndRange: Range1D) startIndex sourceLength (rowOffset: int) (columnOffset: int) (sourceRows: ClArray<int>) (sourceColumns: ClArray<int>) (sourceValues: ClArray<'c>) (resultRows: ClArray<int>) (resultColumns: ClArray<int>) (resultValues: ClArray<'c>) ->

                let gid = ndRange.GlobalID0

                if gid < sourceLength then
                    let index = startIndex + gid

                    resultRows.[index] <- rowOffset + sourceRows.[gid]
                    resultColumns.[index] <- columnOffset + sourceColumns.[gid]
                    resultValues.[index] <- sourceValues.[gid] @>

        let kernel = clContext.Compile <| copyToResult

        fun (processor: MailboxProcessor<_>) startIndex (rowOffset: int) (columnOffset: int) (resultMatrix: COO<'c>) (sourceMatrix: COO<'c>) ->

            let ndRange =
                Range1D.CreateValid(sourceMatrix.NNZ, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(
                Msg.MsgSetArguments
                    (fun () ->
                        kernel.KernelFunc
                            ndRange
                            startIndex
                            sourceMatrix.NNZ
                            rowOffset
                            columnOffset
                            sourceMatrix.Rows
                            sourceMatrix.Columns
                            sourceMatrix.Values
                            resultMatrix.Rows
                            resultMatrix.Columns
                            resultMatrix.Values)
            )

            processor.Post(Msg.CreateRunMsg<_, _> kernel)

    let private insertZero (clContext: ClContext) workGroupSize =

        let copy = copyToResult clContext workGroupSize

        fun queue startIndex (zeroCounts: ClArray<int> array) (matrixZero: COO<'c>) resultMatrix ->

            let rowCount = zeroCounts.Length

            let mutable startIndex = startIndex

            let insertMany row firstColumn count =
                for i in 0 .. count - 1 do
                    let rowOffset = row * matrixZero.RowCount

                    let columnOffset =
                        (firstColumn + i) * matrixZero.ColumnCount

                    copy queue startIndex rowOffset columnOffset resultMatrix matrixZero

                    startIndex <- startIndex + matrixZero.NNZ

            for row in 0 .. rowCount - 1 do
                let zeroCountInRow = zeroCounts.[row].ToHostAndFree queue

                let mutable column = 0

                for count in zeroCountInRow do
                    insertMany row column count

                    column <- column + count + 1

    let private insertNonZero (clContext: ClContext) workGroupSize op =

        let itemTo = ClArray.itemTo clContext workGroupSize

        let preparePositions =
            preparePositions clContext workGroupSize op

        let setPositions = setPositions clContext workGroupSize

        fun queue (rowBoundaries: (int * int) array) (matrixRight: CSR<'b>) (leftValues: ClArray<'a>) (leftColsHost: int array) (resultMatrix: COO<'c>) ->

            let setPositions =
                setPositions queue matrixRight.RowCount matrixRight.ColumnCount

            let rowCount = rowBoundaries.Length

            let length =
                matrixRight.RowCount * matrixRight.ColumnCount

            let bitmap =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, length)

            let mappedMatrix =
                clContext.CreateClArrayWithSpecificAllocationMode<'c>(DeviceOnly, length)

            let mutable startIndex = 0

            let value =
                clContext.CreateClCell Unchecked.defaultof<'a>

            for row in 0 .. rowCount - 1 do
                let leftEdge, rightEdge = rowBoundaries.[row]

                for i in leftEdge .. rightEdge do
                    itemTo queue i leftValues value

                    let column = leftColsHost.[i]

                    let rowOffset = row * matrixRight.RowCount
                    let columnOffset = column * matrixRight.ColumnCount

                    preparePositions queue value matrixRight mappedMatrix bitmap

                    startIndex <- setPositions rowOffset columnOffset startIndex resultMatrix mappedMatrix bitmap

            value.Free queue
            bitmap.Free queue
            mappedMatrix.Free queue

            startIndex

    let private countZeroElements (clContext: ClContext) workGroupSize =

        let countZeroElementsInRow =
            <@ fun (ndRange: Range1D) (firstIndex: int) (lastIndex: int) (columnCount: int) (columns: ClArray<int>) (result: ClArray<int>) ->

                let gid = ndRange.GlobalID0

                let nnzInRow = lastIndex - firstIndex + 1

                if gid <= nnzInRow then

                    if nnzInRow = 0 then
                        result.[0] <- columnCount

                    elif gid = nnzInRow then
                        result.[nnzInRow] <- columnCount - columns.[lastIndex] - 1

                    elif gid = 0 then
                        result.[0] <- columns.[firstIndex]

                    else
                        result.[gid] <-
                            columns.[firstIndex + gid]
                            - columns.[firstIndex + gid - 1]
                            - 1 @>

        let kernel = clContext.Compile countZeroElementsInRow

        fun (queue: MailboxProcessor<_>) (matrix: CSR<_>) (rowBoundaries: (int * int) array) ->

            let kernel = kernel.GetKernel()

            let (zeroCounts: ClArray<int> array) = Array.zeroCreate matrix.RowCount

            for row in 0 .. matrix.RowCount - 1 do

                let firstIndex = fst rowBoundaries.[row]
                let lastIndex = snd rowBoundaries.[row]

                let nnzInRow = lastIndex - firstIndex + 1
                let length = nnzInRow + 1

                let ndRange =
                    Range1D.CreateValid(length, workGroupSize)

                let result =
                    clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, length)

                queue.Post(
                    Msg.MsgSetArguments
                        (fun () ->
                            kernel.KernelFunc ndRange firstIndex lastIndex matrix.ColumnCount matrix.Columns result)
                )

                queue.Post(Msg.CreateRunMsg<_, _>(kernel))

                zeroCounts.[row] <- result

            zeroCounts

    let private mapAll<'a, 'b, 'c when 'a: struct and 'b: struct and 'c: struct and 'c: equality>
        (clContext: ClContext)
        workGroupSize
        (op: Expr<'a option -> 'b option -> 'c option>)
        =

        let pairwise = ClArray.pairwise clContext workGroupSize

        let mapInPlace =
            ClArray.mapInPlace <@ fun (a, b) -> (a, b - 1) @> clContext workGroupSize

        let countZeroElements =
            countZeroElements clContext workGroupSize

        let insertNonZero = insertNonZero clContext workGroupSize op

        let insertZero = insertZero clContext workGroupSize

        fun (queue: MailboxProcessor<_>) allocationMode (resultNNZ: int) (matrixZero: COO<'c> option) (matrixLeft: CSR<'a>) (matrixRight: CSR<'b>) ->

            let resultRows =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(allocationMode, resultNNZ)

            let resultColumns =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(allocationMode, resultNNZ)

            let resultValues =
                clContext.CreateClArrayWithSpecificAllocationMode<'c>(allocationMode, resultNNZ)

            let resultMatrix =
                { Context = clContext
                  Rows = resultRows
                  Columns = resultColumns
                  Values = resultValues
                  RowCount = matrixLeft.RowCount * matrixRight.RowCount
                  ColumnCount = matrixLeft.ColumnCount * matrixRight.ColumnCount }

            let leftColumns = matrixLeft.Columns.ToHost queue

            let pairsOfRowPointers =
                pairwise queue DeviceOnly matrixLeft.RowPointers
                |> Option.defaultWith
                    (fun () -> failwith "The state of the matrix is broken. The length of the rowPointers must be >= 2")

            mapInPlace queue pairsOfRowPointers

            let rowBoundaries = pairsOfRowPointers.ToHostAndFree queue

            let zeroCounts =
                countZeroElements queue matrixLeft rowBoundaries

            let startIndex =
                insertNonZero queue rowBoundaries matrixRight matrixLeft.Values leftColumns resultMatrix

            matrixZero
            |> Option.iter (fun m -> insertZero queue startIndex zeroCounts m resultMatrix)

            resultMatrix

    let run<'a, 'b, 'c when 'a: struct and 'b: struct and 'c: struct and 'c: equality>
        (clContext: ClContext)
        workGroupSize
        (op: Expr<'a option -> 'b option -> 'c option>)
        =

        let getSize =
            getAllocationSize clContext workGroupSize op

        let mapWithValue =
            Map.WithValue.run clContext op workGroupSize

        let mapAll = mapAll clContext workGroupSize op

        let bitonic =
            Common.Sort.Bitonic.sortKeyValuesInplace clContext workGroupSize

        fun (queue: MailboxProcessor<_>) allocationMode (matrixLeft: CSR<'a>) (matrixRight: CSR<'b>) ->

            let matrixZero =
                mapWithValue queue allocationMode None matrixRight

            let size =
                getSize queue matrixZero matrixLeft matrixRight

            if size = 0 then
                matrixZero
                |> Option.iter (fun m -> m.Dispose queue)

                None
            else
                let result =
                    mapAll queue allocationMode size matrixZero matrixLeft matrixRight

                matrixZero
                |> Option.iter (fun m -> m.Dispose queue)

                bitonic queue result.Rows result.Columns result.Values

                result |> Some
