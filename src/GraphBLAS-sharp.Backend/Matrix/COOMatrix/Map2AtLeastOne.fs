namespace GraphBLAS.FSharp.Backend.Matrix.COO

open Brahma.FSharp
open GraphBLAS.FSharp.Backend.Matrix
open GraphBLAS.FSharp.Backend.Quotes
open Microsoft.FSharp.Quotations
open GraphBLAS.FSharp.Backend.Objects
open GraphBLAS.FSharp.Backend
open GraphBLAS.FSharp.Backend.Objects.ClMatrix
open GraphBLAS.FSharp.Backend.Objects.ClContext

module internal Map2AtLeastOne =
    let preparePositionsAtLeastOne<'a, 'b, 'c when 'a: struct and 'b: struct and 'c: struct and 'c: equality>
        (clContext: ClContext)
        (opAdd: Expr<'a option -> 'b option -> 'c option>)
        workGroupSize
        =

        let preparePositions =
            <@ fun (ndRange: Range1D) length (allRowsBuffer: ClArray<int>) (allColumnsBuffer: ClArray<int>) (leftValuesBuffer: ClArray<'a>) (rightValuesBuffer: ClArray<'b>) (allValuesBuffer: ClArray<'c>) (rawPositionsBuffer: ClArray<int>) (isLeftBitmap: ClArray<int>) ->

                let i = ndRange.GlobalID0

                if (i < length - 1
                    && allRowsBuffer.[i] = allRowsBuffer.[i + 1]
                    && allColumnsBuffer.[i] = allColumnsBuffer.[i + 1]) then

                    let result =
                        (%opAdd) (Some leftValuesBuffer.[i + 1]) (Some rightValuesBuffer.[i])

                    (%PreparePositions.both) i result rawPositionsBuffer allValuesBuffer
                elif (i > 0
                      && i < length
                      && (allRowsBuffer.[i] <> allRowsBuffer.[i - 1]
                          || allColumnsBuffer.[i] <> allColumnsBuffer.[i - 1]))
                     || i = 0 then

                    let leftResult =
                        (%opAdd) (Some leftValuesBuffer.[i]) None

                    let rightResult =
                        (%opAdd) None (Some rightValuesBuffer.[i])

                    (%PreparePositions.leftRight)
                        i
                        leftResult
                        rightResult
                        isLeftBitmap
                        allValuesBuffer
                        rawPositionsBuffer @>

        let kernel = clContext.Compile(preparePositions)

        fun (processor: MailboxProcessor<_>) (allRows: ClArray<int>) (allColumns: ClArray<int>) (leftValues: ClArray<'a>) (rightValues: ClArray<'b>) (isLeft: ClArray<int>) ->
            let length = leftValues.Length

            let ndRange =
                Range1D.CreateValid(length, workGroupSize)

            let rawPositionsGpu =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, length)

            let allValues =
                clContext.CreateClArrayWithSpecificAllocationMode<'c>(DeviceOnly, length)

            let kernel = kernel.GetKernel()

            processor.Post(
                Msg.MsgSetArguments
                    (fun () ->
                        kernel.KernelFunc
                            ndRange
                            length
                            allRows
                            allColumns
                            leftValues
                            rightValues
                            allValues
                            rawPositionsGpu
                            isLeft)
            )

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

            rawPositionsGpu, allValues

    let merge<'a, 'b when 'a: struct and 'b: struct> (clContext: ClContext) workGroupSize =

        let merge =
            <@ fun (ndRange: Range1D) firstSide secondSide sumOfSides (firstRowsBuffer: ClArray<int>) (firstColumnsBuffer: ClArray<int>) (firstValuesBuffer: ClArray<'a>) (secondRowsBuffer: ClArray<int>) (secondColumnsBuffer: ClArray<int>) (secondValuesBuffer: ClArray<'b>) (allRowsBuffer: ClArray<int>) (allColumnsBuffer: ClArray<int>) (leftMergedValuesBuffer: ClArray<'a>) (rightMergedValuesBuffer: ClArray<'b>) (isLeftBitmap: ClArray<int>) ->

                let i = ndRange.GlobalID0

                let mutable beginIdxLocal = local ()
                let mutable endIdxLocal = local ()
                let localID = ndRange.LocalID0

                if localID < 2 then
                    let x = localID * (workGroupSize - 1) + i - 1

                    let diagonalNumber = min (sumOfSides - 1) x

                    let mutable leftEdge = diagonalNumber + 1 - secondSide
                    leftEdge <- max 0 leftEdge

                    let mutable rightEdge = firstSide - 1

                    rightEdge <- min diagonalNumber rightEdge

                    while leftEdge <= rightEdge do
                        let middleIdx = (leftEdge + rightEdge) / 2

                        let firstIndex: uint64 =
                            ((uint64 firstRowsBuffer.[middleIdx]) <<< 32)
                            ||| (uint64 firstColumnsBuffer.[middleIdx])

                        let secondIndex: uint64 =
                            ((uint64 secondRowsBuffer.[diagonalNumber - middleIdx])
                             <<< 32)
                            ||| (uint64 secondColumnsBuffer.[diagonalNumber - middleIdx])

                        if firstIndex < secondIndex then
                            leftEdge <- middleIdx + 1
                        else
                            rightEdge <- middleIdx - 1

                    // Here localID equals either 0 or 1
                    if localID = 0 then
                        beginIdxLocal <- leftEdge
                    else
                        endIdxLocal <- leftEdge

                barrierLocal ()

                let beginIdx = beginIdxLocal
                let endIdx = endIdxLocal
                let firstLocalLength = endIdx - beginIdx
                let mutable x = workGroupSize - firstLocalLength

                if endIdx = firstSide then
                    x <- secondSide - i + localID + beginIdx

                let secondLocalLength = x

                //First indices are from 0 to firstLocalLength - 1 inclusive
                //Second indices are from firstLocalLength to firstLocalLength + secondLocalLength - 1 inclusive
                let localIndices = localArray<uint64> workGroupSize

                if localID < firstLocalLength then
                    localIndices.[localID] <-
                        ((uint64 firstRowsBuffer.[beginIdx + localID])
                         <<< 32)
                        ||| (uint64 firstColumnsBuffer.[beginIdx + localID])

                if localID < secondLocalLength then
                    localIndices.[firstLocalLength + localID] <-
                        ((uint64 secondRowsBuffer.[i - beginIdx]) <<< 32)
                        ||| (uint64 secondColumnsBuffer.[i - beginIdx])

                barrierLocal ()

                if i < sumOfSides then
                    let mutable leftEdge = localID + 1 - secondLocalLength
                    leftEdge <- max 0 leftEdge

                    let mutable rightEdge = firstLocalLength - 1

                    rightEdge <- min localID rightEdge

                    while leftEdge <= rightEdge do
                        let middleIdx = (leftEdge + rightEdge) / 2
                        let firstIndex = localIndices.[middleIdx]

                        let secondIndex =
                            localIndices.[firstLocalLength + localID - middleIdx]

                        if firstIndex < secondIndex then
                            leftEdge <- middleIdx + 1
                        else
                            rightEdge <- middleIdx - 1

                    let boundaryX = rightEdge
                    let boundaryY = localID - leftEdge

                    // boundaryX and boundaryY can't be off the right edge of array (only off the left edge)
                    let isValidX = boundaryX >= 0
                    let isValidY = boundaryY >= 0

                    let mutable fstIdx = 0UL

                    if isValidX then
                        fstIdx <- localIndices.[boundaryX]

                    let mutable sndIdx = 0UL

                    if isValidY then
                        sndIdx <- localIndices.[firstLocalLength + boundaryY]

                    if not isValidX || isValidY && fstIdx < sndIdx then
                        allRowsBuffer.[i] <- int (sndIdx >>> 32)
                        allColumnsBuffer.[i] <- int sndIdx
                        rightMergedValuesBuffer.[i] <- secondValuesBuffer.[i - localID - beginIdx + boundaryY]
                        isLeftBitmap.[i] <- 0
                    else
                        allRowsBuffer.[i] <- int (fstIdx >>> 32)
                        allColumnsBuffer.[i] <- int fstIdx
                        leftMergedValuesBuffer.[i] <- firstValuesBuffer.[beginIdx + boundaryX]
                        isLeftBitmap.[i] <- 1 @>

        let kernel = clContext.Compile(merge)

        fun (processor: MailboxProcessor<_>) (matrixLeftRows: ClArray<int>) (matrixLeftColumns: ClArray<int>) (matrixLeftValues: ClArray<'a>) (matrixRightRows: ClArray<int>) (matrixRightColumns: ClArray<int>) (matrixRightValues: ClArray<'b>) ->

            let firstSide = matrixLeftValues.Length
            let secondSide = matrixRightValues.Length
            let sumOfSides = firstSide + secondSide

            let allRows =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, sumOfSides)

            let allColumns =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, sumOfSides)

            let leftMergedValues =
                clContext.CreateClArrayWithSpecificAllocationMode<'a>(DeviceOnly, sumOfSides)

            let rightMergedValues =
                clContext.CreateClArrayWithSpecificAllocationMode<'b>(DeviceOnly, sumOfSides)

            let isLeft =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, sumOfSides)

            let ndRange =
                Range1D.CreateValid(sumOfSides, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(
                Msg.MsgSetArguments
                    (fun () ->
                        kernel.KernelFunc
                            ndRange
                            firstSide
                            secondSide
                            sumOfSides
                            matrixLeftRows
                            matrixLeftColumns
                            matrixLeftValues
                            matrixRightRows
                            matrixRightColumns
                            matrixRightValues
                            allRows
                            allColumns
                            leftMergedValues
                            rightMergedValues
                            isLeft)
            )

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

            allRows, allColumns, leftMergedValues, rightMergedValues, isLeft

    ///<param name="clContext">.</param>
    ///<param name="opAdd">.</param>
    ///<param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let run<'a, 'b, 'c when 'a: struct and 'b: struct and 'c: struct and 'c: equality>
        (clContext: ClContext)
        (opAdd: Expr<'a option -> 'b option -> 'c option>)
        workGroupSize
        =

        let merge = merge clContext workGroupSize

        let preparePositions =
            preparePositionsAtLeastOne clContext opAdd workGroupSize

        let setPositions =
            Common.setPositions<'c> clContext workGroupSize

        fun (queue: MailboxProcessor<_>) allocationMode (matrixLeft: ClMatrix.COO<'a>) (matrixRight: ClMatrix.COO<'b>) ->

            let allRows, allColumns, leftMergedValues, rightMergedValues, isLeft =
                merge
                    queue
                    matrixLeft.Rows
                    matrixLeft.Columns
                    matrixLeft.Values
                    matrixRight.Rows
                    matrixRight.Columns
                    matrixRight.Values

            let rawPositions, allValues =
                preparePositions queue allRows allColumns leftMergedValues rightMergedValues isLeft

            queue.Post(Msg.CreateFreeMsg<_>(leftMergedValues))
            queue.Post(Msg.CreateFreeMsg<_>(rightMergedValues))

            let resultRows, resultColumns, resultValues, _ =
                setPositions queue allocationMode allRows allColumns allValues rawPositions

            queue.Post(Msg.CreateFreeMsg<_>(isLeft))
            queue.Post(Msg.CreateFreeMsg<_>(rawPositions))
            queue.Post(Msg.CreateFreeMsg<_>(allRows))
            queue.Post(Msg.CreateFreeMsg<_>(allColumns))
            queue.Post(Msg.CreateFreeMsg<_>(allValues))

            { Context = clContext
              RowCount = matrixLeft.RowCount
              ColumnCount = matrixLeft.ColumnCount
              Rows = resultRows
              Columns = resultColumns
              Values = resultValues }
