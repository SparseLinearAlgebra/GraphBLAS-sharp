namespace GraphBLAS.FSharp.Backend.Vector.Dense

open Brahma.FSharp
open Microsoft.FSharp.Quotations
open GraphBLAS.FSharp
open GraphBLAS.FSharp.Backend.Common
open GraphBLAS.FSharp.Backend.Quotes
open GraphBLAS.FSharp.Objects.ClVector
open GraphBLAS.FSharp.Objects.ClContextExtensions
open GraphBLAS.FSharp.Objects.ClCellExtensions
open GraphBLAS.FSharp.Objects.ArraysExtensions

module Vector =
    let map<'a, 'b when 'a: struct and 'b: struct>
        (op: Expr<'a option -> 'b option>)
        (clContext: ClContext)
        workGroupSize
        =

        let map = Map.map op clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (leftVector: ClArray<'a option>) ->

            map processor allocationMode leftVector

    let map2InPlace<'a, 'b, 'c when 'a: struct and 'b: struct and 'c: struct>
        (opAdd: Expr<'a option -> 'b option -> 'c option>)
        (clContext: ClContext)
        workGroupSize
        =

        let map2InPlace =
            Map.map2InPlace opAdd clContext workGroupSize

        fun (processor: RawCommandQueue) (leftVector: ClArray<'a option>) (rightVector: ClArray<'b option>) (resultVector: ClArray<'c option>) ->

            map2InPlace processor leftVector rightVector resultVector

    let map2<'a, 'b, 'c when 'a: struct and 'b: struct and 'c: struct>
        (opAdd: Expr<'a option -> 'b option -> 'c option>)
        (clContext: ClContext)
        workGroupSize
        =

        let map2 = Map.map2 opAdd clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (leftVector: ClArray<'a option>) (rightVector: ClArray<'b option>) ->

            map2 processor allocationMode leftVector rightVector

    let map2AtLeastOne op clContext workGroupSize =
        map2 (Convert.atLeastOneToOption op) clContext workGroupSize

    let assignByMaskInPlace<'a, 'b when 'a: struct and 'b: struct>
        (maskOp: Expr<'a option -> 'b option -> 'a -> 'a option>)
        (clContext: ClContext)
        workGroupSize
        =

        let fillSubVectorKernel =
            <@ fun (ndRange: Range1D) resultLength (leftVector: ClArray<'a option>) (maskVector: ClArray<'b option>) (value: ClCell<'a>) (resultVector: ClArray<'a option>) ->

                let gid = ndRange.GlobalID0

                if gid < resultLength then
                    resultVector.[gid] <- (%maskOp) leftVector.[gid] maskVector.[gid] value.Value @>

        let kernel = clContext.Compile(fillSubVectorKernel)

        fun (processor: RawCommandQueue) (leftVector: ClArray<'a option>) (maskVector: ClArray<'b option>) (value: 'a) (resultVector: ClArray<'a option>) ->

            let ndRange =
                Range1D.CreateValid(leftVector.Length, workGroupSize)

            let kernel = kernel.GetKernel()

            let valueCell = clContext.CreateClCell(value)

            kernel.KernelFunc ndRange leftVector.Length leftVector maskVector valueCell resultVector

            processor.RunKernel kernel

            valueCell.Free()

    let assignByMask<'a, 'b when 'a: struct and 'b: struct>
        (maskOp: Expr<'a option -> 'b option -> 'a -> 'a option>)
        (clContext: ClContext)
        workGroupSize
        =

        let assignByMask =
            assignByMaskInPlace maskOp clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (leftVector: ClArray<'a option>) (maskVector: ClArray<'b option>) (value: 'a) ->
            let resultVector =
                clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, leftVector.Length)

            assignByMask processor leftVector maskVector value resultVector

            resultVector

    let assignBySparseMaskInPlace<'a, 'b when 'a: struct and 'b: struct>
        (maskOp: Expr<'a option -> 'b option -> 'a -> 'a option>)
        (clContext: ClContext)
        workGroupSize
        =

        let fillSubVectorKernel =
            <@ fun (ndRange: Range1D) resultLength (leftVector: ClArray<'a option>) (maskVectorIndices: ClArray<int>) (maskVectorValues: ClArray<'b>) (value: ClCell<'a>) (resultVector: ClArray<'a option>) ->

                let gid = ndRange.GlobalID0

                if gid < resultLength then
                    let i = maskVectorIndices.[gid]
                    resultVector.[i] <- (%maskOp) leftVector.[i] (Some maskVectorValues.[gid]) value.Value @>

        let kernel = clContext.Compile(fillSubVectorKernel)

        fun (processor: RawCommandQueue) (leftVector: ClArray<'a option>) (maskVector: Sparse<'b>) (value: 'a) (resultVector: ClArray<'a option>) ->

            let ndRange =
                Range1D.CreateValid(maskVector.NNZ, workGroupSize)

            let kernel = kernel.GetKernel()

            let valueCell = clContext.CreateClCell(value)

            kernel.KernelFunc
                ndRange
                maskVector.NNZ
                leftVector
                maskVector.Indices
                maskVector.Values
                valueCell
                resultVector


            processor.RunKernel kernel


            valueCell.Free()

    // TODO: toSparseUnsorted + bitonic probably would work faster
    let toSparse<'a when 'a: struct> (clContext: ClContext) workGroupSize =
        let scatterValues =
            Common.Scatter.lastOccurrence clContext workGroupSize

        let scatterIndices =
            Common.Scatter.lastOccurrence clContext workGroupSize

        let getBitmap =
            Map.map (Map.option 1 0) clContext workGroupSize

        let prefixSum =
            Common.PrefixSum.standardExcludeInPlace clContext workGroupSize

        let allIndices =
            ClArray.init Map.id clContext workGroupSize

        let allValues =
            Map.map (Map.optionToValueOrZero Unchecked.defaultof<'a>) clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (vector: ClArray<'a option>) ->

            let positions = getBitmap processor DeviceOnly vector

            let resultLength =
                (prefixSum processor positions)
                    .ToHostAndFree(processor)

            // compute result indices
            let resultIndices =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(allocationMode, resultLength)

            let allIndices =
                allIndices processor DeviceOnly vector.Length

            scatterIndices processor positions allIndices resultIndices

            allIndices.Free()

            // compute result values
            let resultValues =
                clContext.CreateClArrayWithSpecificAllocationMode<'a>(allocationMode, resultLength)

            let allValues = allValues processor DeviceOnly vector

            scatterValues processor positions allValues resultValues

            allValues.Free()

            positions.Free()

            { Context = clContext
              Indices = resultIndices
              Values = resultValues
              Size = vector.Length }

    let toSparseUnsorted<'a when 'a: struct> (clContext: ClContext) workGroupSize =

        let kernel =
            <@ fun (ndRange: Range1D) (inputLength: int) (inputValues: ClArray<'a option>) (resultSize: ClCell<int>) (resultIndices: ClArray<int>) (resultValues: ClArray<'a>) ->

                let gid = ndRange.GlobalID0

                if gid < inputLength then
                    match inputValues.[gid] with
                    | Some v ->
                        let offset = atomic (+) resultSize.Value 1
                        resultIndices.[offset] <- gid
                        resultValues.[offset] <- v
                    | None -> () @>

        let kernel = clContext.Compile kernel

        let copy = ClArray.copy clContext workGroupSize
        let copyValues = ClArray.copy clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (vector: ClArray<'a option>) ->

            let tempIndices =
                clContext.CreateClArrayWithSpecificAllocationMode<int>(DeviceOnly, vector.Length)

            let tempValues =
                clContext.CreateClArrayWithSpecificAllocationMode<'a>(DeviceOnly, vector.Length)

            let resultLengthCell = clContext.CreateClCell<int>(0)

            let ndRange =
                Range1D.CreateValid(vector.Length, workGroupSize)

            let kernel = kernel.GetKernel()

            kernel.KernelFunc ndRange vector.Length vector resultLengthCell tempIndices tempValues

            processor.RunKernel kernel

            let resultLength =
                resultLengthCell.ToHostAndFree(processor)

            let resultIndices =
                copy processor allocationMode tempIndices resultLength

            let resultValues =
                copyValues processor allocationMode tempValues resultLength

            tempIndices.Free()
            tempValues.Free()

            { Context = clContext
              Indices = resultIndices
              Values = resultValues
              Size = vector.Length }

    let reduce<'a when 'a: struct> (opAdd: Expr<'a -> 'a -> 'a>) (clContext: ClContext) workGroupSize =

        let choose =
            ClArray.choose Map.id clContext workGroupSize

        let reduce =
            Common.Reduce.reduce opAdd clContext workGroupSize

        fun (processor: RawCommandQueue) (vector: ClArray<'a option>) ->
            choose processor DeviceOnly vector
            |> function
                | Some values ->
                    let result = reduce processor values

                    values.Free()

                    result
                | None -> clContext.CreateClCell Unchecked.defaultof<'a>

    let ofList (clContext: ClContext) workGroupSize =
        let scatter =
            Common.Scatter.lastOccurrence clContext workGroupSize

        let zeroCreate =
            ClArray.zeroCreate clContext workGroupSize

        let map =
            Backend.Common.Map.map <@ Some @> clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode size (elements: (int * 'a) list) ->
            let indices, values = elements |> Array.ofList |> Array.unzip

            let values =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, values)

            let indices =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, indices)

            let mappedValues = map processor DeviceOnly values

            let result = zeroCreate processor allocationMode size

            scatter processor indices mappedValues result

            mappedValues.Free()
            indices.Free()
            values.Free()

            result
