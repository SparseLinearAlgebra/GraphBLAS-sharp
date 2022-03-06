namespace GraphBLAS.FSharp

type VectorFormat = | COO

type Vector<'a when 'a: struct> =
    | VectorCOO of COOVector<'a>

    member this.Size =
        match this with
        | VectorCOO vector -> vector.Size

and COOVector<'a> =
    { mutable Size: int
      mutable Indices: int []
      mutable Values: 'a [] }

    override this.ToString() =
        [ sprintf "Sparse Vector\n"
          sprintf "Size:    %i \n" this.Size
          sprintf "Indices: %A \n" this.Indices
          sprintf "Values:  %A \n" this.Values ]
        |> String.concat ""

    static member FromTuples(size: int, indices: int [], values: 'a []) =
        { Size = size
          Indices = indices
          Values = values }

    static member FromArray(array: 'a [], isZero: 'a -> bool) =
        let (indices, vals) =
            array
            |> Seq.cast<'a>
            |> Seq.mapi (fun idx v -> (idx, v))
            |> Seq.filter (fun (_, v) -> not (isZero v))
            |> Array.ofSeq
            |> Array.unzip

        COOVector.FromTuples(array.Length, indices, vals)

type VectorTuples<'a> = { Indices: int []; Values: 'a [] }
