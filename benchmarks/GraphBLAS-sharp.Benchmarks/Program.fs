open GraphBLAS.FSharp.Benchmarks
open BenchmarkDotNet.Running

[<EntryPoint>]
let main argv =
    let benchmarks =
        BenchmarkSwitcher [| typeof<BFSBenchmarksWithoutDataTransfer>
                             typeof<BFSPushPullBenchmarksWithoutDataTransfer> |]

    benchmarks.Run argv |> ignore
    0
