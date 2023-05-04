﻿namespace GraphBLAS.FSharp.Backend.Quotes

open GraphBLAS.FSharp.Backend.Objects

module ArithmeticOperations =
    let inline mkUnaryOp zero unaryOp =
        <@ fun x ->
            let mutable res = zero

            match x with
            | Some v -> res <- (%unaryOp) v
            | None -> res <- (%unaryOp) zero

            if res = zero then None else Some res @>

    let inline mkNumericSum zero =
        <@ fun (x: 't option) (y: 't option) ->
            let mutable res = zero

            match x, y with
            | Some f, Some s -> res <- f + s
            | Some f, None -> res <- f
            | None, Some s -> res <- s
            | None, None -> ()

            if res = zero then None else Some res @>

    let inline mkNumericSumAsMul zero =
        <@ fun (x: 't option) (y: 't option) ->
            let mutable res = zero

            match x, y with
            | Some f, Some s -> res <- f + s
            | _ -> ()

            if res = zero then None else Some res @>

    let inline mkNumericSumAtLeastOne zero =
        <@ fun (values: AtLeastOne<'t, 't>) ->
            let mutable res = zero

            match values with
            | Both (f, s) -> res <- f + s
            | Left f -> res <- f
            | Right s -> res <- s

            if res = zero then None else Some res @>

    let inline mkNumericMul zero =
        <@ fun (x: 't option) (y: 't option) ->
            let mutable res = zero

            match x, y with
            | Some f, Some s -> res <- f * s
            | _ -> ()

            if res = zero then None else Some res @>

    let inline mkNumericMulAtLeastOne zero =
        <@ fun (values: AtLeastOne<'t, 't>) ->
            let mutable res = zero

            match values with
            | Both (f, s) -> res <- f * s
            | _ -> ()

            if res = zero then None else Some res @>

    let boolSum =
        <@ fun (x: bool option) (y: bool option) ->
            let mutable res = false

            match x, y with
            | None, None -> ()
            | _ -> res <- true

            if res then Some true else None @>

    let inline addLeftConst zero constant =
        mkUnaryOp zero <@ fun x -> constant + x @>

    let inline addRightConst zero constant =
        mkUnaryOp zero <@ fun x -> x + constant @>

    let intSumExplicit =
        mkNumericSumAsMul (System.Int32.MaxValue)

    let intSum = mkNumericSum 0
    let byteSum = mkNumericSum 0uy
    let floatSum = mkNumericSum 0.0
    let float32Sum = mkNumericSum 0f

    let boolSumAtLeastOne =
        <@ fun (_: AtLeastOne<bool, bool>) -> Some true @>

    let intSumAtLeastOne = mkNumericSumAtLeastOne 0
    let byteSumAtLeastOne = mkNumericSumAtLeastOne 0uy
    let floatSumAtLeastOne = mkNumericSumAtLeastOne 0.0
    let float32SumAtLeastOne = mkNumericSumAtLeastOne 0f

    let boolMul =
        <@ fun (x: bool option) (y: bool option) ->
            let mutable res = false

            match x, y with
            | Some _, Some _ -> res <- true
            | _ -> ()

            if res then Some true else None @>

    let inline mulLeftConst zero constant =
        mkUnaryOp zero <@ fun x -> constant * x @>

    let inline mulRightConst zero constant =
        mkUnaryOp zero <@ fun x -> x * constant @>

    let intMul = mkNumericMul 0
    let byteMul = mkNumericMul 0uy
    let floatMul = mkNumericMul 0.0
    let float32Mul = mkNumericMul 0f

    let boolMulAtLeastOne =
        <@ fun (values: AtLeastOne<bool, bool>) ->
            let mutable res = false

            match values with
            | Both _ -> res <- true
            | _ -> ()

            if res then Some true else None @>

    let intMulAtLeastOne = mkNumericMulAtLeastOne 0
    let byteMulAtLeastOne = mkNumericMulAtLeastOne 0uy
    let floatMulAtLeastOne = mkNumericMulAtLeastOne 0.0
    let float32MulAtLeastOne = mkNumericMulAtLeastOne 0f

    let notQ =
        <@ fun x ->
            match x with
            | Some true -> None
            | _ -> Some true @>

    let min<'a when 'a: comparison> =
        <@ fun (x: 'a option) (y: 'a option) ->
            match x, y with
            | Some x, Some y -> Some(min x y)
            | Some x, None -> Some x
            | None, Some y -> Some y
            | _ -> None @>

    let less<'a when 'a: comparison> =
        <@ fun (x: 'a option) (y: 'a option) ->
            match x, y with
            | Some x, Some y -> if (x < y) then Some 1 else None
            | Some x, None -> Some 1
            | _ -> None @>
