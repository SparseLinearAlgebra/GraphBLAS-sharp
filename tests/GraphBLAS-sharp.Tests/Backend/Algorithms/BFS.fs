module GraphBLAS.FSharp.Tests.Backend.Algorithms.BFS

open Expecto
open GraphBLAS.FSharp
open GraphBLAS.FSharp.Backend.Quotes
open GraphBLAS.FSharp.Tests
open GraphBLAS.FSharp.Tests.Context
open GraphBLAS.FSharp.Tests.Backend.QuickGraph.Algorithms
open GraphBLAS.FSharp.Tests.Backend.QuickGraph.CreateGraph
open GraphBLAS.FSharp.Objects.ClVectorExtensions
open GraphBLAS.FSharp.Objects

let testFixtures (testContext: TestContext) =
    [ let config = Utils.undirectedAlgoConfig
      let context = testContext.ClContext
      let queue = testContext.Queue
      let workGroupSize = Constants.Common.defaultWorkGroupSize

      let testName =
          sprintf "Test on %A" testContext.ClContext

      let bfs =
          Algorithms.BFS.singleSource
              ArithmeticOperations.boolSumOption
              ArithmeticOperations.boolMulOption
              context
              workGroupSize

      let bfsSparse =
          Algorithms.BFS.singleSourceSparse
              ArithmeticOperations.boolSumOption
              ArithmeticOperations.boolMulOption
              context
              workGroupSize

      let bfsPushPull =
          Algorithms.BFS.singleSourcePushPull
              ArithmeticOperations.boolSumOption
              ArithmeticOperations.boolMulOption
              context
              workGroupSize

      testPropertyWithConfig config testName
      <| fun (matrix: int [,]) ->

          let matrixBool = Array2D.map (fun x -> x <> 0) matrix

          let graph = undirectedFromArray2D matrix 0

          let largestComponent =
              ConnectedComponents.largestComponent graph

          if largestComponent.Length > 0 then
              let source = largestComponent.[0]

              let expected =
                  (snd (BFS.runUndirected graph source))
                  |> Utils.createArrayFromDictionary (Array2D.length1 matrix) 0

              let matrixHost =
                  Utils.createMatrixFromArray2D CSR matrixBool ((=) false)

              let matrixHostBool =
                  Utils.createMatrixFromArray2D CSR (Array2D.map (fun x -> x <> 0) matrix) ((=) false)

              let matrix = matrixHost.ToDevice context
              let matrixBool = matrixHostBool.ToDevice context

              let res = bfs queue matrix source

              let resSparse = bfsSparse queue matrixBool source

              let resPushPull = bfsPushPull queue matrixBool source

              let resHost = res.ToHost queue
              let resHostSparse = resSparse.ToHost queue
              let resHostPushPull = resPushPull.ToHost queue

              matrix.Dispose()
              matrixBool.Dispose()
              res.Dispose()
              resSparse.Dispose()
              resPushPull.Dispose()

              match resHost, resHostSparse, resHostPushPull with
              | Vector.Dense resHost, Vector.Dense resHostSparse, Vector.Dense resHostPushPull ->
                  let actual = resHost |> Utils.unwrapOptionArray 0

                  let actualSparse =
                      resHostSparse |> Utils.unwrapOptionArray 0

                  let actualPushPull =
                      resHostPushPull |> Utils.unwrapOptionArray 0

                  Expect.sequenceEqual actual expected "Dense bfs is not as expected"
                  Expect.sequenceEqual actualSparse expected "Sparse bfs is not as expected"
                  Expect.sequenceEqual actualPushPull expected "Push-pull bfs is not as expected"
              | _ -> failwith "Not implemented" ]

let tests =
    TestCases.gpuTests "Bfs tests" testFixtures
