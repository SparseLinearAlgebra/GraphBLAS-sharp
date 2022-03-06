module Algo.Bfs

open Expecto
open Expecto.Logging
open Expecto.Logging.Message
open Brahma.FSharp.OpenCL
open GraphBLAS.FSharp.Backend.Common
open GraphBLAS.FSharp
open GraphBLAS.FSharp.IO
open GraphBLAS.FSharp.Algorithms

let logger = Log.create "Bfs.Tests"

let testCases =
    [ ptestCase ""
      <| fun () ->
          let expected =
              graphblas {
                  let! matrix =
                      MtxReader("arc130.mtx").ReadMatrixReal(fun _ -> 1)
                      |> Matrix.switch CSR

                  return!
                      BFS.levelSingleSource matrix 0
                      >>= Vector.synchronizeAndReturn
              }
              |> EvalGB.withClContext (ClContext())
              |> EvalGB.runSync

          Expect.isTrue true "" ]

let tests = testCases |> testList "Bfs tests"
