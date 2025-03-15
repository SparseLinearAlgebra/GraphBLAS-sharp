namespace rec GraphBLAS.FSharp.Benchmarks

namespace GraphBLAS.FSharp.Benchmarks

open Brahma.FSharp
open Brahma.FSharp.OpenCL.Translator
open Brahma.FSharp.OpenCL.Translator.QuotationTransformers
open OpenCL.Net
open System.IO
open System.Text.RegularExpressions
open GraphBLAS.FSharp.Tests
open FsCheck
open Expecto
open System.Runtime.CompilerServices

module Utils =
    type BenchmarkContext =
        { ClContext: Brahma.FSharp.ClContext
          Queue: RawCommandQueue }

    
    let relativePath = System.Environment.GetEnvironmentVariable "BENCH_ROOT_PATH" // Path.Combine  [| whereAmI("") ;"""../../../"""|]
    

    let getMatricesFilenames configFilename =
        let getFullPathToConfig filename =
            Path.Combine [| relativePath
                            "Configs"
                            filename |]
            |> Path.GetFullPath


        configFilename
        |> getFullPathToConfig
        |> File.ReadLines
        |> Seq.filter (fun line -> not <| line.StartsWith "!")

    let getFullPathToMatrix datasetsFolder matrixFilename =
        Path.Combine [| relativePath
                        "Datasets"
                        datasetsFolder
                        matrixFilename |]

    let availableContexts =
        let pathToConfig =
            Path.Combine [| relativePath
                            "Configs"
                            "Context.txt" |]
            |> Path.GetFullPath

        use reader = new StreamReader(pathToConfig)
        let platformRegex = Regex <| reader.ReadLine()

        let deviceType =
            match reader.ReadLine() with
            | "Cpu" -> DeviceType.Cpu
            | "Gpu" -> DeviceType.Gpu
            | "Default" -> DeviceType.Default
            | _ -> failwith "Unsupported"

        let workGroupSizes =
            reader.ReadLine()
            |> (fun s -> s.Split ' ')
            |> Seq.map int

        let mutable e = ErrorCode.Unknown

        let contexts =
            Cl.GetPlatformIDs &e
            |> Array.collect (fun platform -> Cl.GetDeviceIDs(platform, deviceType, &e))
            |> Seq.ofArray
            |> Seq.distinctBy
                (fun device ->
                    Cl
                        .GetDeviceInfo(device, DeviceInfo.Name, &e)
                        .ToString())
            |> Seq.filter
                (fun device ->
                    let platform =
                        Cl
                            .GetDeviceInfo(device, DeviceInfo.Platform, &e)
                            .CastTo<Platform>()

                    let platformName =
                        Cl
                            .GetPlatformInfo(platform, PlatformInfo.Name, &e)
                            .ToString()

                    platformRegex.IsMatch platformName)
            |> Seq.map
                (fun device ->
                    let platform =
                        Cl
                            .GetDeviceInfo(device, DeviceInfo.Platform, &e)
                            .CastTo<Platform>()

                    let clPlatform =
                        Cl
                            .GetPlatformInfo(platform, PlatformInfo.Name, &e)
                            .ToString()
                        |> Platform.Custom

                    let device =
                        ClDevice.GetFirstAppropriateDevice(clPlatform)

                    let translator = FSQuotationToOpenCLTranslator device

                    let context =
                        Brahma.FSharp.ClContext(device, translator)

                    let queue =
                        RawCommandQueue(context.ClDevice.Device, context.Context, context.Translator)

                    { ClContext = context; Queue = queue })

        seq {
            for wgSize in workGroupSizes do
                for context in contexts do
                    yield (context, wgSize)
        }

    let nextSingle (random: System.Random) =
        let buffer = Array.zeroCreate<byte> 4
        random.NextBytes buffer
        System.BitConverter.ToSingle(buffer, 0)

    let normalFloatGenerator =
        (Arb.Default.NormalFloat()
         |> Arb.toGen
         |> Gen.map float)

    let fIsEqual x y =
        abs (x - y) < Accuracy.medium.absolute
        || x.Equals y

    let nextInt (random: System.Random) = random.Next()

module VectorGenerator =
    let private pairOfVectorsOfEqualSize (valuesGenerator: Gen<'a>) createVector =
        gen {
            let! length = Gen.sized <| Gen.constant

            let! leftArray = Gen.arrayOfLength length valuesGenerator

            let! rightArray = Gen.arrayOfLength length valuesGenerator

            return (createVector leftArray, createVector rightArray)
        }

    let intPair format =
        fun array -> Utils.createVectorFromArray format array ((=) 0)
        |> pairOfVectorsOfEqualSize Arb.generate<int32>

    let floatPair format =
        let fIsEqual x y =
            abs (x - y) < Accuracy.medium.absolute || x = y

        let createVector array =
            Utils.createVectorFromArray format array (fIsEqual 0.0)

        pairOfVectorsOfEqualSize Utils.normalFloatGenerator createVector
