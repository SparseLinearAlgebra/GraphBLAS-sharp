namespace GraphBLAS.FSharp.Backend.Matrix.SpGeMM

open Brahma.FSharp
open GraphBLAS.FSharp.Backend.Common
open GraphBLAS.FSharp.Backend.Common.Sort
open GraphBLAS.FSharp.Backend.Matrix
open GraphBLAS.FSharp.Backend.Quotes
open GraphBLAS.FSharp.Backend.Objects.ClContext
open GraphBLAS.FSharp.Backend.Objects.ArraysExtensions
open GraphBLAS.FSharp.Backend.Objects
open GraphBLAS.FSharp.Backend.Objects.ClCell
open FSharp.Quotations
open GraphBLAS.FSharp.Backend.Vector.Sparse
open GraphBLAS.FSharp.Backend.Objects.ClVector

type Indices = ClArray<int>

type Values<'a> = ClArray<'a>

module Expand =
    let getRowsLength (clContext: ClContext) workGroupSize =
        let subtract =
            ClArray.map2 clContext workGroupSize Map.subtraction

        let pairwise = ClArray.pairwise clContext workGroupSize

        fun (processor: MailboxProcessor<_>) (matrix: ClMatrix.CSR<'b>) ->

            let firstPointers, secondPointers =
                pairwise processor DeviceOnly matrix.RowPointers

            let rowsLength = subtract processor DeviceOnly secondPointers firstPointers

            firstPointers.Free processor
            secondPointers.Free processor

            rowsLength

    let getSegmentPointers (clContext: ClContext) workGroupSize =

        let gather = Gather.run clContext workGroupSize

        let prefixSum =
            PrefixSum.standardExcludeInplace clContext workGroupSize

        fun (processor: MailboxProcessor<_>) (leftMatrixRow: ClVector.Sparse<'a>) (rightMatrixRowsLengths: ClArray<int>) ->

            let segmentsLengths =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, leftMatrixRow.Indices.Length)

            // extract needed lengths by left matrix nnz
            gather processor leftMatrixRow.Indices rightMatrixRowsLengths segmentsLengths

            rightMatrixRowsLengths.Free processor

            // compute pointers
            let length =
                (prefixSum processor segmentsLengths)
                    .ToHostAndFree processor

            length, segmentsLengths

    let expand (clContext: ClContext) workGroupSize =

        let idScatter =
            Scatter.initLastOccurrence Map.id clContext workGroupSize

        let scatter =
            Scatter.lastOccurrence clContext workGroupSize

        let zeroCreate =
            ClArray.zeroCreate clContext workGroupSize

        let maxPrefixSum =
            PrefixSum.runIncludeInplace <@ max @> clContext workGroupSize

        let create = ClArray.create clContext workGroupSize

        let gather = Gather.run clContext workGroupSize

        let segmentPrefixSum =
            PrefixSum.ByKey.sequentialInclude clContext workGroupSize <@ (+) @> 0

        let removeDuplicates =
            ClArray.removeDuplications clContext workGroupSize

        let leftMatrixGather = Gather.run clContext workGroupSize

        let rightMatrixGather = Gather.run clContext workGroupSize

        fun (processor: MailboxProcessor<_>) lengths (segmentsPointers: Indices) (leftMatrixRow: ClVector.Sparse<'a>) (rightMatrix: ClMatrix.CSR<'b>) ->

            // Compute left matrix positions
            let leftMatrixPositions = zeroCreate processor DeviceOnly lengths

            idScatter processor segmentsPointers leftMatrixPositions

            (maxPrefixSum processor leftMatrixPositions 0)
                .Free processor

            // Compute right matrix positions
            let rightMatrixPositions = create processor DeviceOnly lengths 1

            let requiredRightMatrixPointers =
                zeroCreate processor DeviceOnly leftMatrixRow.Indices.Length

            gather processor leftMatrixRow.Indices rightMatrix.RowPointers requiredRightMatrixPointers

            scatter processor segmentsPointers requiredRightMatrixPointers rightMatrixPositions

            requiredRightMatrixPointers.Free processor

            // another way to get offsets ???
            let offsets =
                removeDuplicates processor segmentsPointers

            segmentPrefixSum processor offsets.Length rightMatrixPositions leftMatrixPositions offsets

            offsets.Free processor

            // compute columns
            let columns =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, lengths)

            gather processor rightMatrixPositions rightMatrix.Columns columns

            // compute left matrix values
            let leftMatrixValues =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, lengths)

            leftMatrixGather processor leftMatrixPositions leftMatrixRow.Values leftMatrixValues

            leftMatrixPositions.Free processor

            // compute right matrix values
            let rightMatrixValues =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, lengths)

            rightMatrixGather processor rightMatrixPositions rightMatrix.Values rightMatrixValues

            rightMatrixPositions.Free processor

            // left, right matrix values, columns and rows indices
            leftMatrixValues, rightMatrixValues, columns

    let multiply (clContext: ClContext) workGroupSize (predicate: Expr<'a -> 'b -> 'c option>) =
        let getBitmap =
            ClArray.map2<'a, 'b, int> clContext workGroupSize
            <| Map.choose2Bitmap predicate

        let prefixSum =
            PrefixSum.standardExcludeInplace clContext workGroupSize

        let assignValues =
            ClArray.assignOption2 clContext workGroupSize predicate

        let scatter =
            Scatter.lastOccurrence clContext workGroupSize

        fun (processor: MailboxProcessor<_>) (firstValues: ClArray<'a>) (secondValues: ClArray<'b>) (columns: Indices) ->

            let positions =
                getBitmap processor DeviceOnly firstValues secondValues

            let resultLength =
                (prefixSum processor positions)
                    .ToHostAndFree(processor)

            if resultLength = 0 then None
            else
                let resultIndices =
                    clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, resultLength)

                scatter processor positions columns resultIndices

                let resultValues =
                    clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, resultLength)

                assignValues processor firstValues secondValues positions resultValues

                Some (resultValues, resultIndices)

    let sortByColumns (clContext: ClContext) workGroupSize =

        let sortByKeyValues =
            Radix.runByKeysStandard clContext workGroupSize

        let sortKeys =
            Radix.standardRunKeysOnly clContext workGroupSize

        fun (processor: MailboxProcessor<_>) (values: ClArray<'a>) (columns: Indices) ->
            // sort by columns
            let sortedValues =
                sortByKeyValues processor DeviceOnly columns values

            let sortedColumns = sortKeys processor columns

            sortedValues, sortedColumns

    let reduce (clContext: ClContext) workGroupSize opAdd =

        let reduce =
            Reduce.ByKey.Option.segmentSequential clContext workGroupSize opAdd

        let getUniqueBitmap =
            ClArray.getUniqueBitmapLastOccurrence clContext workGroupSize

        let prefixSum =
            PrefixSum.standardExcludeInplace clContext workGroupSize

        let idScatter =
            Scatter.initFirsOccurrence Map.id clContext workGroupSize

        fun (processor: MailboxProcessor<_>) allocationMode (values: ClArray<'a>) (columns: Indices) ->

            let bitmap =
                getUniqueBitmap processor DeviceOnly columns

            let uniqueKeysCount =
                (prefixSum processor bitmap)
                    .ToHostAndFree processor

            let offsets =
                clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, uniqueKeysCount)

            idScatter processor bitmap offsets

            bitmap.Free processor

            let reduceResult = // by size variance TODO()
                reduce processor allocationMode uniqueKeysCount offsets columns values

            offsets.Free processor

            reduceResult

    let runRow (clContext: ClContext) workGroupSize opAdd opMul =
        let getSegmentPointers =
            getSegmentPointers clContext workGroupSize

        let expand = expand clContext workGroupSize

        let multiply = multiply clContext workGroupSize opMul

        let sort = sortByColumns clContext workGroupSize

        let reduce = reduce clContext workGroupSize opAdd

        // left matrix last --- for curring
        fun (processor: MailboxProcessor<_>) allocationMode (rightMatrix: ClMatrix.CSR<'b>) (leftMatrixRowsLengths: Indices) (leftMatrixRow: ClVector.Sparse<'a>) ->
            // TODO(sort in range)
            // required right matrix lengths
            let length, segmentPointers =
                getSegmentPointers processor leftMatrixRow leftMatrixRowsLengths

            // expand
            let leftMatrixValues, rightMatrixValues, columns =
                expand processor length segmentPointers leftMatrixRow rightMatrix

            // multiplication
            let mulResult =
                multiply processor leftMatrixValues rightMatrixValues columns

            leftMatrixValues.Free processor
            rightMatrixValues.Free processor
            columns.Free processor

            // check multiplication result
            mulResult
            |> Option.bind (fun (resultValues, resultColumns) ->
                // sort
                let sortedValues, sortedColumns =
                    sort processor resultValues resultColumns

                resultValues.Free processor
                resultColumns.Free processor

                let reduceResult =
                    reduce processor allocationMode sortedValues sortedColumns

                sortedValues.Free processor
                sortedColumns.Free processor

                // create sparse vector (TODO(empty vector))
                reduceResult
                |> Option.bind (fun (values, columns) ->
                    { Context = clContext
                      Indices = columns
                      Values = values
                      Size = rightMatrix.ColumnCount }
                    |> Some))

    let run<'a, 'b, 'c when 'a : struct and 'b : struct and 'c : struct>
        (clContext: ClContext)
        workGroupSize
        opAdd
        (opMul: Expr<'a -> 'b -> 'c option>) =

        let getRowsLength =
            getRowsLength clContext workGroupSize

        let split = CSR.Matrix.byRowsLazy clContext workGroupSize

        let runRow = runRow clContext workGroupSize opAdd opMul

        fun (processor: MailboxProcessor<_>) allocationMode (leftMatrix: ClMatrix.CSR<'a>) (rightMatrix: ClMatrix.CSR<'b>) ->

            let rightMatrixRowsLengths =
                getRowsLength processor rightMatrix

            let runRow =
                runRow processor allocationMode rightMatrix rightMatrixRowsLengths

            split processor allocationMode leftMatrix
            |> Seq.map (fun lazyRow -> Option.bind runRow lazyRow.Value)
