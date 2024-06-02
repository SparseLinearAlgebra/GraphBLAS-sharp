module GraphBLAS.FSharp.Tests.Backend.Vector.Convert

open Expecto
open Expecto.Logging
open Expecto.Logging.Message
open GraphBLAS.FSharp
open GraphBLAS.FSharp.Tests
open TestCases
open GraphBLAS.FSharp.Backend
open GraphBLAS.FSharp.Objects
open GraphBLAS.FSharp.Objects.ClVectorExtensions
open GraphBLAS.FSharp.Objects.ClContextExtensions
open Brahma.FSharp

let logger =
    Log.create "Backend.Vector.Convert.Tests"

let config = Utils.defaultConfig

let wgSize = Constants.Common.defaultWorkGroupSize

let makeTest
    formatFrom
    (convertFun: RawCommandQueue -> AllocationFlag -> ClVector<'a> -> ClVector<'a>)
    (convertFunUnsorted: option<RawCommandQueue -> AllocationFlag -> ClVector<'a> -> ClVector<'a>>)
    isZero
    case
    (array: 'a [])
    =

    let vector =
        Utils.createVectorFromArray formatFrom array isZero

    if vector.NNZ > 0 then

        let context = case.TestContext.ClContext
        let q = case.TestContext.Queue

        let actual =
            let clVector = vector.ToDevice context
            let convertedVector = convertFun q DeviceOnly clVector

            let res = convertedVector.ToHost q

            clVector.Dispose()
            convertedVector.Dispose()

            res

        logger.debug (
            eventX "Actual is {actual}"
            >> setField "actual" $"%A{actual}"
        )

        let expected =
            Utils.createVectorFromArray case.Format array isZero

        Expect.equal actual expected "Vectors must be the same"

        match convertFunUnsorted with
        | None -> ()
        | Some convertFunUnsorted ->
            let clVector = vector.ToDevice context
            let convertedVector = convertFunUnsorted q DeviceOnly clVector

            let res = convertedVector.ToHost q

            match res, expected with
            | Vector.Sparse res, Vector.Sparse expected ->
                let iv = Array.zip res.Indices res.Values
                let resSorted = Array.sortBy (fun (i, v) -> i) iv
                let indices, values = Array.unzip resSorted
                Expect.equal indices expected.Indices "Indices must be the same"
                Expect.equal values expected.Values "Values must be the same"
                Expect.equal res.Size expected.Size "Size must be the same"
            | _ -> ()

            clVector.Dispose()
            convertedVector.Dispose()

let testFixtures case =
    let getCorrectnessTestName datatype formatFrom =
        sprintf $"Correctness on %s{datatype}, %A{formatFrom} -> %A{case.Format}"

    let context = case.TestContext.ClContext
    let q = case.TestContext.Queue

    //q.Error.Add(fun e -> failwithf "%A" e)

    match case.Format with
    | Sparse ->
        [ let convertFun = Vector.toSparse context wgSize
          let convertFunUnsorted = Vector.toSparseUnsorted context wgSize

          Utils.listOfUnionCases<VectorFormat>
          |> List.map
              (fun formatFrom ->
                  makeTest formatFrom convertFun (Some convertFunUnsorted) ((=) 0) case
                  |> testPropertyWithConfig config (getCorrectnessTestName "int" formatFrom))

          let convertFun = Vector.toSparse context wgSize
          let convertFunUnsorted = Vector.toSparseUnsorted context wgSize

          Utils.listOfUnionCases<VectorFormat>
          |> List.map
              (fun formatFrom ->
                  makeTest formatFrom convertFun (Some convertFunUnsorted) ((=) false) case
                  |> testPropertyWithConfig config (getCorrectnessTestName "bool" formatFrom)) ]
        |> List.concat
    | Dense ->
        [ let convertFun = Vector.toDense context wgSize

          Utils.listOfUnionCases<VectorFormat>
          |> List.map
              (fun formatFrom ->
                  makeTest formatFrom convertFun None ((=) 0) case
                  |> testPropertyWithConfig config (getCorrectnessTestName "int" formatFrom))

          let convertFun = Vector.toDense context wgSize

          Utils.listOfUnionCases<VectorFormat>
          |> List.map
              (fun formatFrom ->
                  makeTest formatFrom convertFun None ((=) false) case
                  |> testPropertyWithConfig config (getCorrectnessTestName "bool" formatFrom)) ]
        |> List.concat

let tests =
    operationGPUTests "Backend.Vector.Convert tests" testFixtures
