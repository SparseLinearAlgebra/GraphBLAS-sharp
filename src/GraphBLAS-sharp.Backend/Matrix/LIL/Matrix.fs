namespace GraphBLAS.FSharp.Backend.Matrix.LIL

open Brahma.FSharp
open GraphBLAS.FSharp
open GraphBLAS.FSharp.Objects
open GraphBLAS.FSharp.Objects.ClMatrix
open GraphBLAS.FSharp.Objects.ClContextExtensions

module Matrix =
    let toCSR (clContext: ClContext) workGroupSize =

        let concatIndices = ClArray.concat clContext workGroupSize

        let concatValues = ClArray.concat clContext workGroupSize

        fun (processor: DeviceCommandQueue<_>) allocationMode (matrix: LIL<'a>) ->

            let rowsPointers =
                matrix.Rows
                |> List.map
                    (function
                    | None -> 0
                    | Some vector -> vector.Values.Length)
                |> List.toArray
                // prefix sum
                |> Array.scan (+) 0
                |> fun pointers -> clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, pointers)

            let valuesByRows, columnsIndicesByRows =
                matrix.Rows
                |> List.choose id
                |> List.map (fun vector -> vector.Values, vector.Indices)
                |> List.unzip

            let values =
                concatValues processor allocationMode valuesByRows

            let columnsIndices =
                concatIndices processor allocationMode columnsIndicesByRows

            { Context = clContext
              RowCount = matrix.RowCount
              ColumnCount = matrix.ColumnCount
              RowPointers = rowsPointers
              Columns = columnsIndices
              Values = values }

    let ofVectors (clContext: ClContext) rowCount columnCount (vectors: ClVector.Sparse<_> option list) =
        { Context = clContext
          RowCount = rowCount
          ColumnCount = columnCount
          Rows = vectors }
