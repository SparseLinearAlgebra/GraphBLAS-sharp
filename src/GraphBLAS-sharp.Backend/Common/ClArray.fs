namespace GraphBLAS.FSharp.Backend.Common

open Brahma.FSharp
open Microsoft.FSharp.Quotations
open GraphBLAS.FSharp.Backend.Objects.ClContext
open GraphBLAS.FSharp.Backend.Objects.ClCell
open GraphBLAS.FSharp.Backend.Quotes

module ClArray =
    let init (clContext: ClContext) workGroupSize (initializer: Expr<int -> 'a>) =

        let init =
            <@ fun (range: Range1D) (outputBuffer: ClArray<'a>) (length: int) ->

                let i = range.GlobalID0

                if i < length then
                    outputBuffer.[i] <- (%initializer) i @>

        let program = clContext.Compile(init)

        fun (processor: MailboxProcessor<_>) allocationMode (length: int) ->
            let outputArray =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, length)

            let kernel = program.GetKernel()

            let ndRange =
                Range1D.CreateValid(length, workGroupSize)

            processor.Post(Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange outputArray length))
            processor.Post(Msg.CreateRunMsg<_, _> kernel)

            outputArray

    let create (clContext: ClContext) workGroupSize =

        let create =
            <@ fun (range: Range1D) (outputBuffer: ClArray<'a>) (length: int) (value: ClCell<'a>) ->

                let i = range.GlobalID0

                if i < length then
                    outputBuffer.[i] <- value.Value @>

        let program = clContext.Compile(create)

        fun (processor: MailboxProcessor<_>) allocationMode (length: int) (value: 'a) ->
            let value = clContext.CreateClCell(value)

            let outputArray =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, length)

            let kernel = program.GetKernel()

            let ndRange =
                Range1D.CreateValid(length, workGroupSize)

            processor.Post(Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange outputArray length value))
            processor.Post(Msg.CreateRunMsg<_, _> kernel)
            processor.Post(Msg.CreateFreeMsg(value))

            outputArray

    let zeroCreate<'a> (clContext: ClContext) workGroupSize =

        let create = create clContext workGroupSize

        fun (processor: MailboxProcessor<_>) allocationMode length ->
            create processor allocationMode length Unchecked.defaultof<'a>

    let copy (clContext: ClContext) workGroupSize =
        let copy =
            <@ fun (ndRange: Range1D) (inputArrayBuffer: ClArray<'a>) (outputArrayBuffer: ClArray<'a>) inputArrayLength ->

                let i = ndRange.GlobalID0

                if i < inputArrayLength then
                    outputArrayBuffer.[i] <- inputArrayBuffer.[i] @>

        let program = clContext.Compile(copy)

        fun (processor: MailboxProcessor<_>) allocationMode (inputArray: ClArray<'a>) ->
            let ndRange =
                Range1D.CreateValid(inputArray.Length, workGroupSize)

            let outputArray =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, inputArray.Length)

            let kernel = program.GetKernel()

            processor.Post(
                Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange inputArray outputArray inputArray.Length)
            )

            processor.Post(Msg.CreateRunMsg<_, _> kernel)

            outputArray

    let replicate (clContext: ClContext) workGroupSize =

        let replicate =
            <@ fun (ndRange: Range1D) (inputArrayBuffer: ClArray<'a>) (outputArrayBuffer: ClArray<'a>) inputArrayLength outputArrayLength ->

                let i = ndRange.GlobalID0

                if i < outputArrayLength then
                    outputArrayBuffer.[i] <- inputArrayBuffer.[i % inputArrayLength] @>

        let kernel = clContext.Compile(replicate)

        fun (processor: MailboxProcessor<_>) allocationMode (inputArray: ClArray<'a>) count ->
            let outputArrayLength = inputArray.Length * count

            let outputArray =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, outputArrayLength)

            let ndRange =
                Range1D.CreateValid(outputArray.Length, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(
                Msg.MsgSetArguments
                    (fun () -> kernel.KernelFunc ndRange inputArray outputArray inputArray.Length outputArrayLength)
            )

            processor.Post(Msg.CreateRunMsg<_, _> kernel)

            outputArray

    let getUniqueBitmap (clContext: ClContext) workGroupSize =

        let getUniqueBitmap =
            <@ fun (ndRange: Range1D) (inputArray: ClArray<'a>) inputLength (isUniqueBitmap: ClArray<int>) ->

                let i = ndRange.GlobalID0

                if i < inputLength - 1
                   && inputArray.[i] = inputArray.[i + 1] then
                    isUniqueBitmap.[i] <- 0
                else
                    isUniqueBitmap.[i] <- 1 @>

        let kernel = clContext.Compile(getUniqueBitmap)

        fun (processor: MailboxProcessor<_>) allocationMode (inputArray: ClArray<'a>) ->

            let inputLength = inputArray.Length

            let ndRange =
                Range1D.CreateValid(inputLength, workGroupSize)

            let bitmap =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, inputLength)

            let kernel = kernel.GetKernel()

            processor.Post(Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange inputArray inputLength bitmap))

            processor.Post(Msg.CreateRunMsg<_, _> kernel)

            bitmap

    let getPositions (clContext: ClContext) workGroupSize =

        let getUniqueBitmap = getUniqueBitmap clContext workGroupSize

        let prefixSumExclude =
            PrefixSum.runExcludeInplace <@ (+) @> clContext workGroupSize

        fun (processor: MailboxProcessor<_>) (inputArray: ClArray<'a>) ->

            let bitmap =
                getUniqueBitmap processor DeviceOnly inputArray

            let resultLength =
                (prefixSumExclude processor bitmap 0)
                    .ToHostAndFree(processor)

            bitmap, resultLength

    ///<description>Remove duplicates form the given array.</description>
    ///<param name="clContext">Computational context</param>
    ///<param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    ///<param name="inputArray">Should be sorted.</param>
    let removeDuplications (clContext: ClContext) workGroupSize =

        let scatter =
            Scatter.runInplace clContext workGroupSize

        let getPositions = getPositions clContext workGroupSize

        fun (processor: MailboxProcessor<_>) (inputArray: ClArray<'a>) ->

            let positions, resultLength = getPositions processor inputArray

            let outputArray =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, resultLength)

            scatter processor positions inputArray outputArray

            processor.Post <| Msg.CreateFreeMsg<_>(positions)

            outputArray

    let exists (clContext: ClContext) workGroupSize (predicate: Expr<'a -> bool>) =

        let exists =
            <@ fun (ndRange: Range1D) length (vector: ClArray<'a>) (result: ClCell<bool>) ->

                let gid = ndRange.GlobalID0

                if gid < length then
                    let isExist = (%predicate) vector.[gid]

                    if isExist then result.Value <- true @>

        let kernel = clContext.Compile exists

        fun (processor: MailboxProcessor<_>) (vector: ClArray<'a>) ->

            let result = clContext.CreateClCell false

            let ndRange =
                Range1D.CreateValid(vector.Length, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange vector.Length vector result))

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

            result

    let map<'a, 'b> (clContext: ClContext) workGroupSize (op: Expr<'a -> 'b>) =

        let map =
            <@ fun (ndRange: Range1D) lenght (inputArray: ClArray<'a>) (result: ClArray<'b>) ->

                let gid = ndRange.GlobalID0

                if gid < lenght then
                    result.[gid] <- (%op) inputArray.[gid] @>

        let kernel = clContext.Compile map

        fun (processor: MailboxProcessor<_>) allocationMode (inputArray: ClArray<'a>) ->

            let result =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, inputArray.Length)

            let ndRange =
                Range1D.CreateValid(inputArray.Length, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange inputArray.Length inputArray result))

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

            result

    let map2Inplace<'a, 'b, 'c> (clContext: ClContext) workGroupSize (map: Expr<'a -> 'b -> 'c>) =

        let kernel =
            <@ fun (ndRange: Range1D) length (leftArray: ClArray<'a>) (rightArray: ClArray<'b>) (resultArray: ClArray<'c>) ->

                let gid = ndRange.GlobalID0

                if gid < length then

                    resultArray.[gid] <- (%map) leftArray.[gid] rightArray.[gid] @>

        let kernel = clContext.Compile kernel

        fun (processor: MailboxProcessor<_>) (leftArray: ClArray<'a>) (rightArray: ClArray<'b>) (resultArray: ClArray<'c>) ->

            let ndRange =
                Range1D.CreateValid(resultArray.Length, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(
                Msg.MsgSetArguments
                    (fun () -> kernel.KernelFunc ndRange resultArray.Length leftArray rightArray resultArray)
            )

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

    let map2<'a, 'b, 'c> (clContext: ClContext) workGroupSize map =
        let map2 =
            map2Inplace<'a, 'b, 'c> clContext workGroupSize map

        fun (processor: MailboxProcessor<_>) allocationMode (leftArray: ClArray<'a>) (rightArray: ClArray<'b>) ->

            let resultArray =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, leftArray.Length)

            map2 processor leftArray rightArray resultArray

            resultArray

    let choose<'a, 'b> (clContext: ClContext) workGroupSize (predicate: Expr<'a -> 'b option>) =
        let getBitmap =
            map<'a, int> clContext workGroupSize
            <| Map.chooseBitmap predicate

        let getOptionValues =
            map<'a, 'b option> clContext workGroupSize predicate

        let getValues =
            map<'b option, 'b> clContext workGroupSize
            <| Map.optionToValueOrZero Unchecked.defaultof<'b>

        let prefixSum =
            PrefixSum.runExcludeInplace <@ (+) @> clContext workGroupSize

        let scatter =
            Scatter.runInplace clContext workGroupSize

        fun (processor: MailboxProcessor<_>) allocationMode (array: ClArray<'a>) ->

            let positions = getBitmap processor DeviceOnly array

            let resultLength =
                (prefixSum processor positions 0)
                    .ToHostAndFree(processor)

            let optionValues =
                getOptionValues processor DeviceOnly array

            let values =
                getValues processor DeviceOnly optionValues

            let result =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, resultLength)

            scatter processor positions values result

            result

    let filterOption<'a> (clContext: ClContext) workGroupSize (predicate: Expr<'a option -> bool>) =
        let getBitmap =
            map<'a option, int> clContext workGroupSize
            <| Map.bitmapOption predicate

        let prefixSum =
            PrefixSum.runExcludeInplace <@ (+) @> clContext workGroupSize

        let scatter =
            Scatter.runWithKeys clContext workGroupSize

        fun (processor: MailboxProcessor<_>) allocationMode (array: ClArray<'a option>) (keys: ClArray<int>) ->

            let positions = getBitmap processor DeviceOnly array

            let resultLength =
                (prefixSum processor positions 0)
                    .ToHostAndFree(processor)

            let resultValues, resultKeys =
                scatter processor positions array keys resultLength

            resultKeys, resultValues

    let count (clContext: ClContext) workGroupSize (predicate: Expr<'a -> bool>) =

        let sum =
            PrefixSum.standardExcludeInplace clContext workGroupSize

        let getBitmap =
            <@ fun (ndRange: Range1D) length (vector: ClArray<'a>) (bitmap: ClArray<int>) ->

                let gid = ndRange.GlobalID0

                if gid < length then
                    let isTrue = (%predicate) vector.[gid]

                    if isTrue then
                        bitmap.[gid] <- 1
                    else
                        bitmap.[gid] <- 0 @>

        let kernel = clContext.Compile getBitmap

        fun (processor: MailboxProcessor<_>) (array: ClArray<'a>) ->

            let bitmap =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, array.Length)

            let ndRange =
                Range1D.CreateValid(array.Length, workGroupSize)

            let kernel = kernel.GetKernel()

            processor.Post(Msg.MsgSetArguments(fun () -> kernel.KernelFunc ndRange array.Length array bitmap))

            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

            let result =
                (sum processor bitmap).ToHostAndFree processor

            processor.Post(Msg.CreateFreeMsg bitmap)

            result
