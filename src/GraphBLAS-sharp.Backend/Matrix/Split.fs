namespace GraphBLAS.FSharp.Backend.Matrix

open Brahma.FSharp
open GraphBLAS.FSharp.Backend.Common
open GraphBLAS.FSharp.Backend.Objects
open GraphBLAS.FSharp.Backend.Objects.ClMatrix
open GraphBLAS.FSharp.Backend.Objects.ArraysExtensions
open GraphBLAS.FSharp.Backend.Objects.ClVector

// type lazy matrix ???

module Split =
    module ByChunk =
        let runCOOLazy (clContext: ClContext) workGroupSize =

            let chunkBySizeValues = ClArray.lazyChunkBySize clContext workGroupSize

            let chunkBySizeIndices = ClArray.lazyChunkBySize clContext workGroupSize

            fun (processor: MailboxProcessor<_>) allocationMode chunkSize (matrix: ClMatrix.COO<'a>) ->

               let createSubMatrixLazy (values: Lazy<_>) (columns: Lazy<_>) (rows: Lazy<_>) =
                   lazy
                   { Context = clContext
                     RowCount = matrix.RowCount
                     ColumnCount = matrix.ColumnCount
                     Rows = rows.Value
                     Columns = columns.Value
                     Values = values.Value }

               let values = chunkBySizeValues processor allocationMode chunkSize matrix.Values
               let columns = chunkBySizeIndices processor allocationMode chunkSize matrix.Columns
               let rows = chunkBySizeIndices processor allocationMode chunkSize matrix.Rows

               Seq.map3 createSubMatrixLazy values columns rows

        let runCOO (clContext: ClContext) workGroupSize =

            let run = runCOOLazy clContext workGroupSize

            fun (processor: MailboxProcessor<_>) allocationMode chunkSize (matrix: ClMatrix.COO<'a>) ->
               run processor allocationMode chunkSize matrix
               |> Seq.map (fun lazyMatrix -> lazyMatrix.Value)
               |> Seq.toArray

        // let run (clContext: ClContext) workGroupSize =
        //
        //     let run = runCOOLazy clContext workGroupSize
        //
        //     let runCOO = runCOO clContext workGroupSize
        //
        //     let COOToCSR = COO.Matrix.toCSR clCOntext workGroupSize
        //
        //     fun (processor: MailboxProcessor<_>) allocationMode chunkSize (matrix: ClMatrix<'a>) ->
        //        match matrix with
        //        | ClMatrix.COO matrix -> runCOO processor allocationMode chunkSize matrix
        //        | ClMatrix.COO matrix ->
        //            ()
    module ByRow =
        // MB We can split CSR to chunks without COO representation
        let runCSRLazy (clContext: ClContext) workGroupSize =

            let getChunkValues = ClArray.getChunk clContext workGroupSize

            let getChunkIndices = ClArray.getChunk clContext workGroupSize

            fun (processor: MailboxProcessor<_>) allocationMode (matrix: ClMatrix.CSR<'a>) ->

                let getChunkValues = getChunkValues processor allocationMode matrix.Values
                let getChunkIndices = getChunkIndices processor allocationMode matrix.Columns

                let creatSparseVector values columns =
                    { Context = clContext
                      Indices = columns
                      Values = values
                      Size = matrix.ColumnCount }

                matrix.RowPointers.ToHost processor
                |> Seq.pairwise
                |> Seq.map (fun (first, second) ->
                    lazy
                        if second - first > 0 then
                            let values = getChunkValues first second
                            let columns = getChunkIndices first second

                            Some <| creatSparseVector values columns
                        else None)

        let runCSR (clContext: ClContext) workGroupSize =

            let runLazy = runCSRLazy clContext workGroupSize

            fun (processor: MailboxProcessor<_>) allocationMode (matrix: ClMatrix.CSR<'a>) ->
                runLazy processor allocationMode matrix
                |> Seq.map (fun lazyValue -> lazyValue.Value)
                |> Seq.toArray

    module ByColumn =
        let runCSRLazy (clContext: ClContext) workGroupSize =

            let getChunkValues = ClArray.getChunk clContext workGroupSize

            let getChunkIndices = ClArray.getChunk clContext workGroupSize

            fun (processor: MailboxProcessor<_>) allocationMode (matrix: ClMatrix.CSC<'a>) ->

                let getChunkValues = getChunkValues processor allocationMode matrix.Values
                let getChunkIndices = getChunkIndices processor allocationMode matrix.Rows

                let creatSparseVector values columns =
                    { Context = clContext
                      Indices = columns
                      Values = values
                      Size = matrix.RowCount }

                matrix.ColumnPointers.ToHost processor
                |> Seq.pairwise
                |> Seq.map (fun (first, second) ->
                    lazy
                        if second - first > 0 then
                            let values = getChunkValues first second
                            let rows = getChunkIndices first second

                            Some <| creatSparseVector values rows
                        else None)

        let runCSR (clContext: ClContext) workGroupSize =

            let runLazy = runCSRLazy clContext workGroupSize

            fun (processor: MailboxProcessor<_>) allocationMode (matrix: ClMatrix.CSC<'a>) ->
                runLazy processor allocationMode matrix
                |> Seq.map (fun lazyValue -> lazyValue.Value)
                |> Seq.toArray


