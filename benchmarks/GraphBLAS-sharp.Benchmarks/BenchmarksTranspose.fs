namespace GraphBLAS.FSharp.Benchmarks

open GraphBLAS.FSharp
open GraphBLAS.FSharp.Algorithms
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Columns
open System.IO
open System
open System.Text.RegularExpressions
open Brahma.FSharp.OpenCL
open OpenCL.Net
open GraphBLAS.FSharp.IO

[<Config(typeof<CommonConfig>)>]
type TransposeBenchmarks() =
    let mutable matrix = Unchecked.defaultof<Matrix<float>>

    //TODO fix me
    (*
    [<ParamsSource("AvaliableContextsProvider")>]
    member val OclContext = Unchecked.defaultof<ClContext> with get, set
    member this.Context =
        let (ClContext context) = this.OclContext
        context

    [<ParamsSource("InputMatricesProvider")>]
    member val InputMatrixReader = Unchecked.defaultof<MtxReader> with get, set

    [<GlobalSetup>]
    member this.BuildMatrix() =
        let inputMatrix = this.InputMatrixReader.ReadMatrixReal(float)

        matrix <-
            graphblas {
                return! Matrix.switch CSR inputMatrix
                >>= Matrix.synchronizeAndReturn
            }
            |> EvalGB.withClContext this.Context
            |> EvalGB.runSync

    [<Benchmark>]
    member this.Transpose() =
        Matrix.transpose matrix
        |> EvalGB.withClContext this.Context
        |> EvalGB.runSync

    [<IterationCleanup>]
    member this.ClearBuffers() =
        this.Context.Provider.CloseAllBuffers()

    [<GlobalCleanup>]
    member this.ClearContext() =
        let (ClContext context) = this.OclContext
        context.Provider.Dispose()

    static member AvaliableContextsProvider = Utils.avaliableContexts

    static member InputMatricesProvider =
        "Common.txt"
        |> Utils.getMatricesFilenames
        |> Seq.map
            (fun matrixFilename ->
                match Path.GetExtension matrixFilename with
                | ".mtx" -> MtxReader(Utils.getFullPathToMatrix "Common" matrixFilename)
                | _ -> failwith "Unsupported matrix format"
            )
*)