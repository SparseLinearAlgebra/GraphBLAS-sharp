namespace GraphBLAS.FSharp.Backend

open Brahma.FSharp.OpenCL
open GraphBLAS.FSharp.Backend
open GraphBLAS.FSharp.Backend.Common
open GraphBLAS.FSharp.Backend.Predefined

module internal rec Converter =
    let toCOO
        (clContext: ClContext)
        workGroupSize
        (processor: MailboxProcessor<_>)
        (matrix: CSRMatrix<'a>) =
        let copy = ClArray.copy clContext
        let copyData = ClArray.copy clContext

        let nnz = clContext.CreateClArray(1)

        let rowIndices = prepareRows clContext workGroupSize processor matrix
        PrefixSum.standardIncludeInplace clContext workGroupSize processor rowIndices nnz
        |> ignore

        processor.Post(Msg.CreateFreeMsg(nnz))

        let colIndices =
            copy processor workGroupSize matrix.Columns

        let values =
            copyData processor workGroupSize matrix.Values

        {
            RowCount = matrix.RowCount
            ColumnCount = matrix.ColumnCount
            Rows = rowIndices
            Columns = colIndices
            Values = values
        }

    let private prepareRows
        (clContext: ClContext)
        workGroupSize
        (processor: MailboxProcessor<_>)
        (matrix: CSRMatrix<'a>) =

        let rows = ClArray.create clContext workGroupSize processor matrix.Columns.Length 0

        let rowPointersLength = matrix.RowPointers.Length

        let prepareRows =
            <@
                fun (ndRange: Range1D)
                    (rowPointers: ClArray<int>)
                    (rows: ClArray<int>) ->

                    let i = ndRange.GlobalID0 + 1
                    if i < rowPointersLength - 1 then atomic (+) rows.[rowPointers.[i]] 1 |> ignore
            @>

        if rowPointersLength > 2 then
            let kernel = clContext.CreateClKernel(prepareRows)
            let ndRange = Range1D.CreateValid(rowPointersLength - 2, workGroupSize)
            processor.Post(
                Msg.MsgSetArguments
                    (fun () ->
                        kernel.ArgumentsSetter
                            ndRange
                            matrix.RowPointers
                            rows)
            )
            processor.Post(Msg.CreateRunMsg<_, _>(kernel))

        rows
