﻿namespace GraphBLAS.FSharp.Backend.Common

open Brahma.FSharp
open GraphBLAS.FSharp.Objects.ClContextExtensions
open GraphBLAS.FSharp.Objects.ArraysExtensions
open GraphBLAS.FSharp.Backend.Quotes
open GraphBLAS.FSharp.Backend.Common.Map

module Bitmap =
    let private getUniqueBitmapGeneral predicate (clContext: ClContext) workGroupSize =

        let getUniqueBitmap =
            <@ fun (ndRange: Range1D) (inputArray: ClArray<'a>) inputLength (isUniqueBitmap: ClArray<int>) ->

                let gid = ndRange.GlobalID0

                if gid < inputLength then
                    let isUnique = (%predicate) gid inputLength inputArray // brahma error

                    if isUnique then
                        isUniqueBitmap.[gid] <- 1
                    else
                        isUniqueBitmap.[gid] <- 0 @>

        let kernel = clContext.Compile(getUniqueBitmap)

        fun (processor: RawCommandQueue) allocationMode (inputArray: ClArray<'a>) ->

            let inputLength = inputArray.Length

            let ndRange =
                Range1D.CreateValid(inputLength, workGroupSize)

            let bitmap =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, inputLength)

            let kernel = kernel.GetKernel()

            kernel.KernelFunc ndRange inputArray inputLength bitmap

            processor.RunKernel kernel

            bitmap

    /// <summary>
    /// Gets the bitmap that indicates the first elements of the sequences of consecutive identical elements
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    let firstOccurrence clContext =
        getUniqueBitmapGeneral
        <| Predicates.firstOccurrence ()
        <| clContext

    /// <summary>
    /// Gets the bitmap that indicates the last elements of the sequences of consecutive identical elements
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    let lastOccurrence clContext =
        getUniqueBitmapGeneral
        <| Predicates.lastOccurrence ()
        <| clContext

    let private getUniqueBitmap2General<'a when 'a: equality> getUniqueBitmap (clContext: ClContext) workGroupSize =

        let map =
            map2 <@ fun x y -> x ||| y @> clContext workGroupSize

        let firstGetBitmap = getUniqueBitmap clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (firstArray: ClArray<'a>) (secondArray: ClArray<'a>) ->
            let firstBitmap =
                firstGetBitmap processor DeviceOnly firstArray

            let secondBitmap =
                firstGetBitmap processor DeviceOnly secondArray

            let result =
                map processor allocationMode firstBitmap secondBitmap

            firstBitmap.Free()
            secondBitmap.Free()

            result

    /// <summary>
    /// Gets the bitmap that indicates the first elements of the sequences
    /// of consecutive identical elements from either first array or second array.
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    let firstOccurrence2 clContext =
        getUniqueBitmap2General firstOccurrence clContext

    /// <summary>
    /// Gets the bitmap that indicates the last elements of the sequences
    /// of consecutive identical elements from either first array or second array.
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    let lastOccurrence2 clContext =
        getUniqueBitmap2General lastOccurrence clContext
