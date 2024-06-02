namespace GraphBLAS.FSharp.Backend.Common.Sort

open Brahma.FSharp
open GraphBLAS.FSharp.Backend

module Bitonic =

    let sortRowsColumnsValuesInplace<'a> (clContext: ClContext) (workGroupSize: int) =

        let localSize =
            Common.Utils.floorToPower2 (
                int (clContext.ClDevice.LocalMemSize)
                / (sizeof<uint64> + sizeof<'a>)
            )
            / 2

        let maxThreadsPerBlock =
            min (clContext.ClDevice.MaxWorkGroupSize) (localSize / 2)

        let waveSize = 32
        let maxWorkGroupSize = clContext.ClDevice.MaxWorkGroupSize

        let localStep =
            <@ fun (ndRange: Range1D) (rows: ClArray<int>) (cols: ClArray<int>) (vals: ClArray<'a>) (length: int) ->
                let gid = ndRange.GlobalID0
                let lid = ndRange.LocalID0
                let workGroupSize = ndRange.LocalWorkSize
                let groupId = gid / workGroupSize

                let offset = groupId * localSize
                let border = min (offset + localSize) length
                let n = border - offset

                let nAligned =
                    (%Quotes.ArithmeticOperations.ceilToPowerOfTwo) n

                let numberOfThreads = nAligned / 2

                let sortedKeys = localArray<uint64> localSize
                let sortedVals = localArray<'a> localSize

                let mutable i = lid

                while i + offset < border do
                    let key: uint64 =
                        ((uint64 rows.[i + offset]) <<< 32)
                        ||| (uint64 cols.[i + offset])

                    sortedKeys.[i] <- key
                    sortedVals.[i] <- vals.[i + offset]
                    i <- i + workGroupSize

                barrierLocal ()

                let mutable segmentSize = 2

                while segmentSize <= nAligned do
                    let segmentSizeHalf = segmentSize / 2

                    let mutable tid = lid

                    while tid < numberOfThreads do
                        let segmentId = tid / segmentSizeHalf
                        let innerId = tid % segmentSizeHalf
                        let innerIdSibling = segmentSize - innerId - 1
                        let i = segmentId * segmentSize + innerId
                        let j = segmentId * segmentSize + innerIdSibling

                        if (i < n && j < n && sortedKeys.[i] > sortedKeys.[j]) then
                            let tempK = sortedKeys.[i]
                            sortedKeys.[i] <- sortedKeys.[j]
                            sortedKeys.[j] <- tempK
                            let tempV = sortedVals.[i]
                            sortedVals.[i] <- sortedVals.[j]
                            sortedVals.[j] <- tempV

                        tid <- tid + workGroupSize

                    barrierLocal ()

                    let mutable k = segmentSizeHalf / 2

                    while k > 0 do

                        let mutable tid = lid

                        while tid < numberOfThreads do
                            let segmentSizeInner = k * 2
                            let segmentId = tid / k
                            let innerId = tid % k
                            let innerIdSibling = innerId + k
                            let i = segmentId * segmentSizeInner + innerId

                            let j =
                                segmentId * segmentSizeInner + innerIdSibling

                            if (i < n && j < n && sortedKeys.[i] > sortedKeys.[j]) then
                                let tempK = sortedKeys.[i]
                                sortedKeys.[i] <- sortedKeys.[j]
                                sortedKeys.[j] <- tempK
                                let tempV = sortedVals.[i]
                                sortedVals.[i] <- sortedVals.[j]
                                sortedVals.[j] <- tempV

                            tid <- tid + workGroupSize

                        k <- k / 2
                        barrierLocal ()

                    segmentSize <- segmentSize * 2

                let mutable i = lid

                while i + offset < border do
                    let key = sortedKeys.[i]
                    rows.[i + offset] <- int (key >>> 32)
                    cols.[i + offset] <- int key
                    vals.[i + offset] <- sortedVals.[i]
                    i <- i + workGroupSize @>

        let globalStep =
            <@ fun (ndRange: Range1D) (rows: ClArray<int>) (cols: ClArray<int>) (vals: ClArray<'a>) (length: int) (segmentStart: int) ->
                let lid = ndRange.LocalID0
                let workGroupSize = ndRange.LocalWorkSize

                let n = length

                let nAligned =
                    (%Quotes.ArithmeticOperations.ceilToPowerOfTwo) n

                let numberOfThreads = nAligned / 2

                let mutable segmentSize = segmentStart

                while segmentSize <= nAligned do
                    let segmentSizeHalf = segmentSize / 2

                    let mutable tid = lid

                    while tid < numberOfThreads do
                        let segmentId = tid / segmentSizeHalf
                        let innerId = tid % segmentSizeHalf
                        let innerIdSibling = segmentSize - innerId - 1
                        let i = segmentId * segmentSize + innerId
                        let j = segmentId * segmentSize + innerIdSibling

                        if (i < n && j < n) then
                            let keyI =
                                ((uint64 rows.[i]) <<< 32) ||| (uint64 cols.[i])

                            let keyJ =
                                ((uint64 rows.[j]) <<< 32) ||| (uint64 cols.[j])

                            if (keyI > keyJ) then
                                let tempR = rows.[i]
                                rows.[i] <- rows.[j]
                                rows.[j] <- tempR
                                let tempC = cols.[i]
                                cols.[i] <- cols.[j]
                                cols.[j] <- tempC
                                let tempV = vals.[i]
                                vals.[i] <- vals.[j]
                                vals.[j] <- tempV

                        tid <- tid + workGroupSize

                    barrierGlobal ()

                    let mutable k = segmentSizeHalf / 2

                    while k > 0 do

                        let mutable tid = lid

                        while tid < numberOfThreads do
                            let segmentSizeInner = k * 2
                            let segmentId = tid / k
                            let innerId = tid % k
                            let innerIdSibling = innerId + k
                            let i = segmentId * segmentSizeInner + innerId

                            let j =
                                segmentId * segmentSizeInner + innerIdSibling

                            if (i < n && j < n) then
                                let keyI =
                                    ((uint64 rows.[i]) <<< 32) ||| (uint64 cols.[i])

                                let keyJ =
                                    ((uint64 rows.[j]) <<< 32) ||| (uint64 cols.[j])

                                if (keyI > keyJ) then
                                    let tempR = rows.[i]
                                    rows.[i] <- rows.[j]
                                    rows.[j] <- tempR
                                    let tempC = cols.[i]
                                    cols.[i] <- cols.[j]
                                    cols.[j] <- tempC
                                    let tempV = vals.[i]
                                    vals.[i] <- vals.[j]
                                    vals.[j] <- tempV

                            tid <- tid + workGroupSize

                        k <- k / 2
                        barrierGlobal ()

                    segmentSize <- segmentSize * 2 @>

        let localStep = clContext.Compile(localStep)
        let globalStep = clContext.Compile(globalStep)

        fun (queue: RawCommandQueue) (rows: ClArray<int>) (cols: ClArray<int>) (values: ClArray<'a>) ->

            let size = values.Length

            if (size = 1) then
                ()
            else if (size <= localSize) then
                let numberOfThreads =
                    Common.Utils.ceilToMultiple waveSize (min size maxThreadsPerBlock)

                let ndRangeLocal =
                    Range1D.CreateValid(numberOfThreads, numberOfThreads)

                let kernel = localStep.GetKernel()

                kernel.KernelFunc ndRangeLocal rows cols values values.Length
                queue.RunKernel(kernel)
            else
                let numberOfGroups =
                    size / localSize
                    + (if size % localSize = 0 then 0 else 1)

                let ndRangeLocal =
                    Range1D.CreateValid(maxThreadsPerBlock * numberOfGroups, maxThreadsPerBlock)

                let kernelLocal = localStep.GetKernel()

                kernelLocal.KernelFunc ndRangeLocal rows cols values values.Length

                queue.RunKernel(kernelLocal)

                let ndRangeGlobal =
                    Range1D.CreateValid(maxWorkGroupSize, maxWorkGroupSize)

                let kernelGlobal = globalStep.GetKernel()

                kernelGlobal.KernelFunc ndRangeGlobal rows cols values values.Length (localSize * 2)

                queue.RunKernel(kernelGlobal)


    let sortKeyValuesInplace<'a> (clContext: ClContext) (workGroupSize: int) =

        let localSize =
            Common.Utils.floorToPower2 (
                int (clContext.ClDevice.LocalMemSize)
                / (sizeof<int> + sizeof<'a>)
            )
            / 2

        let maxThreadsPerBlock =
            min (clContext.ClDevice.MaxWorkGroupSize) (localSize / 2)

        let waveSize = 32
        let maxWorkGroupSize = clContext.ClDevice.MaxWorkGroupSize

        let localStep =
            <@ fun (ndRange: Range1D) (rows: ClArray<int>) (vals: ClArray<'a>) (length: int) ->
                let gid = ndRange.GlobalID0
                let lid = ndRange.LocalID0
                let workGroupSize = ndRange.LocalWorkSize
                let groupId = gid / workGroupSize

                let offset = groupId * localSize
                let border = min (offset + localSize) length
                let n = border - offset

                let nAligned =
                    (%Quotes.ArithmeticOperations.ceilToPowerOfTwo) n

                let numberOfThreads = nAligned / 2

                let sortedKeys = localArray<int> localSize
                let sortedVals = localArray<'a> localSize

                let mutable i = lid

                while i + offset < border do
                    let key = rows.[i + offset]
                    sortedKeys.[i] <- key
                    sortedVals.[i] <- vals.[i + offset]
                    i <- i + workGroupSize

                barrierLocal ()

                let mutable segmentSize = 2

                while segmentSize <= nAligned do
                    let segmentSizeHalf = segmentSize / 2

                    let mutable tid = lid

                    while tid < numberOfThreads do
                        let segmentId = tid / segmentSizeHalf
                        let innerId = tid % segmentSizeHalf
                        let innerIdSibling = segmentSize - innerId - 1
                        let i = segmentId * segmentSize + innerId
                        let j = segmentId * segmentSize + innerIdSibling

                        if (i < n && j < n && sortedKeys.[i] > sortedKeys.[j]) then
                            let tempK = sortedKeys.[i]
                            sortedKeys.[i] <- sortedKeys.[j]
                            sortedKeys.[j] <- tempK
                            let tempV = sortedVals.[i]
                            sortedVals.[i] <- sortedVals.[j]
                            sortedVals.[j] <- tempV

                        tid <- tid + workGroupSize

                    barrierLocal ()

                    let mutable k = segmentSizeHalf / 2

                    while k > 0 do

                        let mutable tid = lid

                        while tid < numberOfThreads do
                            let segmentSizeInner = k * 2
                            let segmentId = tid / k
                            let innerId = tid % k
                            let innerIdSibling = innerId + k
                            let i = segmentId * segmentSizeInner + innerId

                            let j =
                                segmentId * segmentSizeInner + innerIdSibling

                            if (i < n && j < n && sortedKeys.[i] > sortedKeys.[j]) then
                                let tempK = sortedKeys.[i]
                                sortedKeys.[i] <- sortedKeys.[j]
                                sortedKeys.[j] <- tempK
                                let tempV = sortedVals.[i]
                                sortedVals.[i] <- sortedVals.[j]
                                sortedVals.[j] <- tempV

                            tid <- tid + workGroupSize

                        k <- k / 2
                        barrierLocal ()

                    segmentSize <- segmentSize * 2

                let mutable i = lid

                while i + offset < border do
                    let key = sortedKeys.[i]
                    rows.[i + offset] <- key
                    vals.[i + offset] <- sortedVals.[i]
                    i <- i + workGroupSize @>

        let globalStep =
            <@ fun (ndRange: Range1D) (rows: ClArray<int>) (vals: ClArray<'a>) (length: int) (segmentStart: int) ->
                let lid = ndRange.LocalID0
                let workGroupSize = ndRange.LocalWorkSize

                let n = length

                let nAligned =
                    (%Quotes.ArithmeticOperations.ceilToPowerOfTwo) n

                let numberOfThreads = nAligned / 2

                let mutable segmentSize = segmentStart

                while segmentSize <= nAligned do
                    let segmentSizeHalf = segmentSize / 2

                    let mutable tid = lid

                    while tid < numberOfThreads do
                        let segmentId = tid / segmentSizeHalf
                        let innerId = tid % segmentSizeHalf
                        let innerIdSibling = segmentSize - innerId - 1
                        let i = segmentId * segmentSize + innerId
                        let j = segmentId * segmentSize + innerIdSibling

                        if (i < n && j < n) then
                            let keyI = rows.[i]
                            let keyJ = rows.[j]

                            if (keyI > keyJ) then
                                let tempR = rows.[i]
                                rows.[i] <- rows.[j]
                                rows.[j] <- tempR
                                let tempV = vals.[i]
                                vals.[i] <- vals.[j]
                                vals.[j] <- tempV

                        tid <- tid + workGroupSize

                    barrierGlobal ()

                    let mutable k = segmentSizeHalf / 2

                    while k > 0 do

                        let mutable tid = lid

                        while tid < numberOfThreads do
                            let segmentSizeInner = k * 2
                            let segmentId = tid / k
                            let innerId = tid % k
                            let innerIdSibling = innerId + k
                            let i = segmentId * segmentSizeInner + innerId

                            let j =
                                segmentId * segmentSizeInner + innerIdSibling

                            if (i < n && j < n) then
                                let keyI = rows.[i]
                                let keyJ = rows.[j]

                                if (keyI > keyJ) then
                                    let tempR = rows.[i]
                                    rows.[i] <- rows.[j]
                                    rows.[j] <- tempR
                                    let tempV = vals.[i]
                                    vals.[i] <- vals.[j]
                                    vals.[j] <- tempV

                            tid <- tid + workGroupSize

                        k <- k / 2
                        barrierGlobal ()

                    segmentSize <- segmentSize * 2 @>

        let localStep = clContext.Compile(localStep)
        let globalStep = clContext.Compile(globalStep)

        fun (queue: RawCommandQueue) (rows: ClArray<int>) (values: ClArray<'a>) ->

            let size = values.Length

            if (size = 1) then
                ()
            else if (size <= localSize) then
                let numberOfThreads =
                    Common.Utils.ceilToMultiple waveSize (min size maxThreadsPerBlock)

                let ndRangeLocal =
                    Range1D.CreateValid(numberOfThreads, numberOfThreads)

                let kernel = localStep.GetKernel()

                kernel.KernelFunc ndRangeLocal rows values values.Length
                queue.RunKernel(kernel)
            else
                let numberOfGroups =
                    size / localSize
                    + (if size % localSize = 0 then 0 else 1)

                let ndRangeLocal =
                    Range1D.CreateValid(maxThreadsPerBlock * numberOfGroups, maxThreadsPerBlock)

                let kernelLocal = localStep.GetKernel()

                kernelLocal.KernelFunc ndRangeLocal rows values values.Length

                queue.RunKernel(kernelLocal)

                let ndRangeGlobal =
                    Range1D.CreateValid(maxWorkGroupSize, maxWorkGroupSize)

                let kernelGlobal = globalStep.GetKernel()

                kernelGlobal.KernelFunc ndRangeGlobal rows values values.Length (localSize * 2)

                queue.RunKernel(kernelGlobal)
