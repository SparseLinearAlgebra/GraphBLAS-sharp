namespace GraphBLAS.FSharp.Backend.Common

open Brahma.FSharp
open GraphBLAS.FSharp.Backend.Quotes
open Microsoft.FSharp.Control
open Microsoft.FSharp.Quotations
open GraphBLAS.FSharp.Backend.Objects.ClContext
open GraphBLAS.FSharp.Backend.Objects.ArraysExtensions
open GraphBLAS.FSharp.Backend.Objects.ClCell

module Reduce =
    /// <summary>
    /// Generalized reduction pattern.
    /// </summary>
    let private runGeneral (clContext: ClContext) workGroupSize scan scanToCell =

        fun (processor: MailboxProcessor<_>) (inputArray: ClArray<'a>) ->

            let scan = scan processor

            let firstLength =
                (inputArray.Length - 1) / workGroupSize + 1

            let firstVerticesArray =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, firstLength)

            let secondLength = (firstLength - 1) / workGroupSize + 1

            let secondVerticesArray =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, secondLength)

            let mutable verticesArrays = firstVerticesArray, secondVerticesArray
            let swap (a, b) = (b, a)

            scan inputArray inputArray.Length (fst verticesArrays)

            let mutable verticesLength = firstLength

            while verticesLength > workGroupSize do
                let fstVertices = fst verticesArrays
                let sndVertices = snd verticesArrays

                scan fstVertices verticesLength sndVertices

                verticesArrays <- swap verticesArrays
                verticesLength <- (verticesLength - 1) / workGroupSize + 1

            let fstVertices = fst verticesArrays

            let result =
                scanToCell processor fstVertices verticesLength

            firstVerticesArray.Free processor
            secondVerticesArray.Free processor

            result

    let private scanSum (clContext: ClContext) (workGroupSize: int) (opAdd: Expr<'a -> 'a -> 'a>) zero =

        let subSum = SubSum.sequentialSum opAdd

        let scan =
            <@ fun (ndRange: Range1D) length (inputArray: ClArray<'a>) (resultArray: ClArray<'a>) ->

                let gid = ndRange.GlobalID0
                let lid = ndRange.LocalID0

                let localValues = localArray<'a> workGroupSize

                if gid < length then
                    localValues.[lid] <- inputArray.[gid]
                else
                    localValues.[lid] <- zero

                barrierLocal ()

                (%subSum) workGroupSize lid localValues

                resultArray.[gid / workGroupSize] <- localValues.[0] @>

        let kernel = clContext.Compile(scan)

        fun (processor: MailboxProcessor<_>) (valuesArray: ClArray<'a>) valuesLength (resultArray: ClArray<'a>) ->
            let ndRange =
                Range1D.CreateValid(valuesArray.Length, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(
                Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange valuesLength valuesArray resultArray)
            )

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

    let private scanToCellSum (clContext: ClContext) workGroupSize (opAdd: Expr<'a -> 'a -> 'a>) zero =

        let subSum = SubSum.sequentialSum opAdd

        let scan =
            <@ fun (ndRange: Range1D) length (inputArray: ClArray<'a>) (resultCell: ClCell<'a>) ->

                let gid = ndRange.GlobalID0
                let lid = ndRange.LocalID0

                let localValues = localArray<'a> workGroupSize

                if gid < length then
                    localValues.[lid] <- inputArray.[gid]
                else
                    localValues.[lid] <- zero

                barrierLocal ()

                (%subSum) workGroupSize lid localValues

                resultCell.Value <- localValues.[0] @>

        let kernel = clContext.Compile(scan)

        fun (processor: MailboxProcessor<_>) (valuesArray: ClArray<'a>) valuesLength ->

            let ndRange =
                Range1D.CreateValid(valuesArray.Length, workGroupSize)

            let resultCell = clContext.CreateClCell zero

            let kernel = kernel.GetKernel()

            processor.Post(Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange valuesLength valuesArray resultCell))

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

            resultCell

    /// <summary>
    /// Summarize array elements.
    /// </summary>
    /// <param name="clContext">ClContext.</param>
    /// <param name="workGroupSize">Work group size.</param>
    /// <param name="op">Summation operation.</param>
    /// <param name="zero">Neutral element for summation.</param>
    let sum (clContext: ClContext) workGroupSize op zero =

        let scan = scanSum clContext workGroupSize op zero

        let scanToCell =
            scanToCellSum clContext workGroupSize op zero

        let run =
            runGeneral clContext workGroupSize scan scanToCell

        fun (processor: MailboxProcessor<_>) (array: ClArray<'a>) -> run processor array

    let private scanReduce<'a when 'a: struct>
        (clContext: ClContext)
        (workGroupSize: int)
        (opAdd: Expr<'a -> 'a -> 'a>)
        =

        let scan =
            <@ fun (ndRange: Range1D) length (inputArray: ClArray<'a>) (resultArray: ClArray<'a>) ->

                let gid = ndRange.GlobalID0
                let lid = ndRange.LocalID0

                let localValues = localArray<'a> workGroupSize

                if gid < length then
                    localValues.[lid] <- inputArray.[gid]

                barrierLocal ()

                if gid < length then

                    (%SubReduce.run opAdd) length workGroupSize gid lid localValues

                    if lid = 0 then
                        resultArray.[gid / workGroupSize] <- localValues.[0] @>

        let kernel = clContext.Compile(scan)

        fun (processor: MailboxProcessor<_>) (valuesArray: ClArray<'a>) valuesLength (resultArray: ClArray<'a>) ->

            let ndRange =
                Range1D.CreateValid(valuesArray.Length, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(
                Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange valuesLength valuesArray resultArray)
            )

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

    let private scanToCellReduce<'a when 'a: struct>
        (clContext: ClContext)
        (workGroupSize: int)
        (opAdd: Expr<'a -> 'a -> 'a>)
        =

        let scan =
            <@ fun (ndRange: Range1D) length (inputArray: ClArray<'a>) (resultValue: ClCell<'a>) ->

                let gid = ndRange.GlobalID0
                let lid = ndRange.LocalID0

                let localValues = localArray<'a> workGroupSize

                if gid < length then
                    localValues.[lid] <- inputArray.[gid]

                barrierLocal ()

                if gid < length then

                    (%SubReduce.run opAdd) length workGroupSize gid lid localValues

                    if lid = 0 then
                        resultValue.Value <- localValues.[0] @>

        let kernel = clContext.Compile(scan)

        fun (processor: MailboxProcessor<_>) (valuesArray: ClArray<'a>) valuesLength ->

            let ndRange =
                Range1D.CreateValid(valuesArray.Length, workGroupSize)

            let resultCell =
                clContext.CreateClCell Unchecked.defaultof<'a>

            let kernel = kernel.GetKernel()

            processor.Post(Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange valuesLength valuesArray resultCell))

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

            resultCell

    /// <summary>
    /// Reduce an array of values.
    /// </summary>
    /// <param name="clContext">ClContext.</param>
    /// <param name="workGroupSize">Work group size.</param>
    /// <param name="op">Reduction operation.</param>
    let reduce (clContext: ClContext) workGroupSize op =

        let scan = scanReduce clContext workGroupSize op

        let scanToCell =
            scanToCellReduce clContext workGroupSize op

        let run =
            runGeneral clContext workGroupSize scan scanToCell

        fun (processor: MailboxProcessor<_>) (array: ClArray<'a>) -> run processor array

    /// <summary>
    /// Reduction of an array of values by an array of keys.
    /// </summary>
    module ByKey =
        /// <summary>
        /// Reduce an array of values by key using a single work item.
        /// </summary>
        /// <param name="clContext">ClContext.</param>
        /// <param name="workGroupSize">Work group size.</param>
        /// <param name="reduceOp">Operation for reducing values.</param>
        /// <remarks>
        /// The length of the result must be calculated in advance.
        /// </remarks>
        let sequential (clContext: ClContext) workGroupSize (reduceOp: Expr<'a -> 'a -> 'a>) =

            let kernel =
                <@ fun (ndRange: Range1D) length (keys: ClArray<int>) (values: ClArray<'a>) (reducedValues: ClArray<'a>) (reducedKeys: ClArray<int>) ->

                    let gid = ndRange.GlobalID0

                    if gid = 0 then
                        let mutable currentKey = keys.[0]
                        let mutable segmentResult = values.[0]
                        let mutable segmentCount = 0

                        for i in 1 .. length - 1 do
                            if currentKey = keys.[i] then
                                segmentResult <- (%reduceOp) segmentResult values.[i]
                            else
                                reducedValues.[segmentCount] <- segmentResult
                                reducedKeys.[segmentCount] <- currentKey

                                segmentCount <- segmentCount + 1
                                currentKey <- keys.[i]
                                segmentResult <- values.[i]

                        reducedKeys.[segmentCount] <- currentKey
                        reducedValues.[segmentCount] <- segmentResult @>

            let kernel = clContext.Compile kernel

            fun (processor: MailboxProcessor<_>) allocationMode (resultLength: int) (keys: ClArray<int>) (values: ClArray<'a>) ->

                let reducedValues =
                    clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, resultLength)

                let reducedKeys =
                    clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, resultLength)

                let ndRange =
                    Range1D.CreateValid(resultLength, workGroupSize)

                let kernel = kernel.GetKernel()

                processor.Post(
                    Msg.MsgSetArguments
                        (fun () -> kernel.KernelFunc ndRange keys.Length keys values reducedValues reducedKeys)
                )

                processor.Post(Msg.CreateRunMsg<_, _>(kernel))

                reducedKeys, reducedValues

        /// <summary>
        /// Reduces values by key. Each segment is reduced by one work item.
        /// </summary>
        /// <param name="clContext">ClContext.</param>
        /// <param name="workGroupSize">Work group size.</param>
        /// <param name="reduceOp">Operation for reducing values.</param>
        /// <remarks>
        /// The length of the result must be calculated in advance.
        /// </remarks>
        let segmentSequential (clContext: ClContext) workGroupSize (reduceOp: Expr<'a -> 'a -> 'a>) =

            let kernel =
                <@ fun (ndRange: Range1D) uniqueKeyCount keysLength (offsets: ClArray<int>) (keys: ClArray<int>) (values: ClArray<'a>) (reducedValues: ClArray<'a>) (reducedKeys: ClArray<int>) ->

                    let gid = ndRange.GlobalID0

                    if gid < uniqueKeyCount then
                        let startPosition = offsets.[gid]

                        let sourceKey = keys.[startPosition]
                        let mutable sum = values.[startPosition]

                        let mutable currentPosition = startPosition + 1

                        while currentPosition < keysLength
                              && sourceKey = keys.[currentPosition] do

                            sum <- (%reduceOp) sum values.[currentPosition]
                            currentPosition <- currentPosition + 1

                        reducedValues.[gid] <- sum
                        reducedKeys.[gid] <- sourceKey @>

            let kernel = clContext.Compile kernel

            fun (processor: MailboxProcessor<_>) allocationMode (resultLength: int) (offsets: ClArray<int>) (keys: ClArray<int>) (values: ClArray<'a>) ->

                let reducedValues =
                    clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, resultLength)

                let reducedKeys =
                    clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, resultLength)

                let ndRange =
                    Range1D.CreateValid(resultLength, workGroupSize)

                let kernel = kernel.GetKernel()

                processor.Post(
                    Msg.MsgSetArguments
                        (fun () ->
                            kernel.KernelFunc
                                ndRange
                                resultLength
                                keys.Length
                                offsets
                                keys
                                values
                                reducedValues
                                reducedKeys)
                )

                processor.Post(Msg.CreateRunMsg<_, _>(kernel))

                reducedKeys, reducedValues

        /// <summary>
        /// Reduces values by key. One work group participates in the reduction.
        /// </summary>
        /// <param name="clContext">ClContext.</param>
        /// <param name="workGroupSize">Work group size.</param>
        /// <param name="reduceOp">Operation for reducing values.</param>
        /// <remarks>
        /// Reduces an array of values that does not exceed the size of the workgroup.
        /// The length of the result must be calculated in advance.
        /// </remarks>
        let oneWorkGroupSegments (clContext: ClContext) workGroupSize (reduceOp: Expr<'a -> 'a -> 'a>) =

            let kernel =
                <@ fun (ndRange: Range1D) length (keys: ClArray<int>) (values: ClArray<'a>) (reducedValues: ClArray<'a>) (reducedKeys: ClArray<int>) ->

                    let lid = ndRange.GlobalID0

                    // load values to local memory (may be without it)
                    let localValues = localArray<'a> workGroupSize

                    if lid < length then
                        localValues.[lid] <- values.[lid]

                    // load keys to local memory (mb without it)
                    let localKeys = localArray<int> workGroupSize

                    if lid < length then
                        localKeys.[lid] <- keys.[lid]

                    // get unique keys bitmap
                    let localBitmap = localArray<int> workGroupSize
                    localBitmap.[lid] <- 0
                    (%PreparePositions.getUniqueBitmapLocal<int>) localKeys workGroupSize lid localBitmap

                    // get positions from bitmap by prefix sum
                    // ??? get bitmap by prefix sum in another kernel ???
                    // ??? we can restrict prefix sum for 0 .. length ???
                    (%SubSum.localIntPrefixSum) lid workGroupSize localBitmap

                    let uniqueKeysCount = localBitmap.[length - 1]

                    if lid < uniqueKeysCount then
                        let itemKeyId = lid + 1

                        let startKeyIndex =
                            (%Search.Bin.lowerPositionLocal) length itemKeyId localBitmap

                        match startKeyIndex with
                        | Some startPosition ->
                            let sourceKeyPosition = localBitmap.[startPosition]
                            let mutable currentSum = localValues.[startPosition]
                            let mutable currentIndex = startPosition + 1

                            while currentIndex < length
                                  && localBitmap.[currentIndex] = sourceKeyPosition do

                                currentSum <- (%reduceOp) currentSum localValues.[currentIndex]
                                currentIndex <- currentIndex + 1

                            reducedKeys.[lid] <- localKeys.[startPosition]
                            reducedValues.[lid] <- currentSum
                        | None -> () @>

            let kernel = clContext.Compile kernel

            fun (processor: MailboxProcessor<_>) allocationMode (resultLength: int) (keys: ClArray<int>) (values: ClArray<'a>) ->
                if keys.Length > workGroupSize then
                    failwith "The length of the value should not exceed the size of the workgroup"

                let reducedValues =
                    clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, resultLength)

                let reducedKeys =
                    clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, resultLength)

                let ndRange =
                    Range1D.CreateValid(resultLength, workGroupSize)

                let kernel = kernel.GetKernel()

                processor.Post(
                    Msg.MsgSetArguments
                        (fun () -> kernel.KernelFunc ndRange keys.Length keys values reducedValues reducedKeys)
                )

                processor.Post(Msg.CreateRunMsg<_, _>(kernel))

                reducedKeys, reducedValues

        let segmentSequentialOption opAdd (clContext: ClContext) workGroupSize =

            let kernel =
                <@ fun (ndRange: Range1D) length uniqueKeysCount (values: ClArray<'a option>) (keys: ClArray<int>) (offsets: ClArray<int>) (resultKeys: ClArray<int>) (resultValues: ClArray<'a option>) ->

                    //Thread per unique key
                    //gid is a serial number of key
                    let gid = ndRange.GlobalID0

                    if gid < uniqueKeysCount then

                        let startIndex =
                            (%Search.Bin.lowerPosition) length gid offsets

                        match startIndex with
                        | Some startIndex ->
                            let key = keys.[startIndex]
                            let mutable sum = None
                            let mutable currentIndex = startIndex

                            while currentIndex < length && keys.[currentIndex] = key do
                                let res = (%opAdd) sum values.[currentIndex]
                                sum <- res
                                currentIndex <- currentIndex + 1

                            resultValues.[gid] <- sum
                            resultKeys.[gid] <- key
                        | None -> () @>

            let computeOffsets =
                ClArray.getPositions clContext workGroupSize

            let kernel = clContext.Compile kernel

            fun (processor: MailboxProcessor<_>) (allocationMode: AllocationFlag) (values: ClArray<'a option>) (keys: ClArray<int>) ->

                let offsets, uniqueKeysCount = computeOffsets processor keys
                let kernel = kernel.GetKernel()

                let ndRange =
                    Range1D.CreateValid(uniqueKeysCount, workGroupSize)

                let resKeys =
                    clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, uniqueKeysCount)

                let resValues =
                    clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, uniqueKeysCount)

                processor.Post(
                    Msg.MsgSetArguments
                        (fun () ->
                            kernel.KernelFunc
                                ndRange
                                values.Length
                                uniqueKeysCount
                                values
                                keys
                                offsets
                                resKeys
                                resValues)
                )

                processor.Post(Msg.CreateRunMsg<_, _> kernel)

                offsets.Free processor

                resKeys, resValues
