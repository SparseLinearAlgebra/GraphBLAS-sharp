namespace GraphBLAS.FSharp

open Brahma.FSharp
open Microsoft.FSharp.Control
open Microsoft.FSharp.Core
open Microsoft.FSharp.Quotations
open GraphBLAS.FSharp
open GraphBLAS.FSharp.Backend
open GraphBLAS.FSharp.Objects
open GraphBLAS.FSharp.Objects.ClContextExtensions
open GraphBLAS.FSharp.Objects.ClVector
open GraphBLAS.FSharp.Backend.Quotes
open GraphBLAS.FSharp.Backend.Vector

[<RequireQualifiedAccess>]
module Vector =
    /// <summary>
    /// Builds vector of given format with fixed size and fills it with the given value.
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let create (clContext: ClContext) workGroupSize =
        let create = ClArray.create clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode size format value ->
            match format with
            | Sparse -> failwith "Attempting to create full sparse vector"
            | Dense ->
                ClVector.Dense
                <| create processor allocationMode size value

    /// <summary>
    /// Builds vector of given format with fixed size and fills it with the default values of desired type.
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let zeroCreate (clContext: ClContext) workGroupSize =
        let create = create clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode size format -> create processor allocationMode size format None

    /// <summary>
    /// Builds vector of given format with fixed size and fills it with the values from the given list.
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let ofList (clContext: ClContext) workGroupSize =
        let scatter =
            Common.Scatter.lastOccurrence clContext workGroupSize

        let zeroCreate =
            ClArray.zeroCreate clContext workGroupSize

        let map =
            Common.Map.map <@ Some @> clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode format size (elements: (int * 'a) list) ->
            match format with
            | Sparse ->
                let indices, values =
                    elements
                    |> Array.ofList
                    |> Array.sortBy fst
                    |> Array.unzip

                { Context = clContext
                  Indices = clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, indices)
                  Values = clContext.CreateClArrayWithSpecificAllocationMode(allocationMode, values)
                  Size = size }
                |> ClVector.Sparse
            | Dense ->
                let indices, values = elements |> Array.ofList |> Array.unzip

                let values =
                    clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, values)

                let indices =
                    clContext.CreateClArrayWithSpecificAllocationMode(DeviceOnly, indices)

                let mappedValues = map processor DeviceOnly values

                let result = zeroCreate processor allocationMode size

                scatter processor indices mappedValues result

                mappedValues.Dispose()
                indices.Dispose()
                values.Dispose()

                ClVector.Dense result

    /// <summary>
    /// Creates new vector with the values from the given one.
    /// New vector represented in the format of the given one.
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let copy (clContext: ClContext) workGroupSize =
        let sparseCopy =
            Sparse.Vector.copy clContext workGroupSize

        let copyOptionData = ClArray.copy clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (vector: ClVector<'a>) ->
            match vector with
            | ClVector.Sparse vector ->
                ClVector.Sparse
                <| sparseCopy processor allocationMode vector
            | ClVector.Dense vector ->
                ClVector.Dense
                <| copyOptionData processor allocationMode vector vector.Length

    /// <summary>
    /// Sparsifies the given vector if it is in a dense format.
    /// If the given vector is already sparse, copies it.
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let toSparse (clContext: ClContext) workGroupSize =
        let toSparse =
            Dense.Vector.toSparse clContext workGroupSize

        let copy = copy clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (vector: ClVector<'a>) ->
            match vector with
            | ClVector.Dense vector ->
                ClVector.Sparse
                <| toSparse processor allocationMode vector
            | ClVector.Sparse _ -> copy processor allocationMode vector

    /// <summary>
    /// Sparsifies the given vector if it is in a dense format.
    /// If the given vector is already sparse, copies it.
    /// Works faster than regular version, but indices of the sparse vector are unsorted.
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let toSparseUnsorted (clContext: ClContext) workGroupSize =
        let toSparse =
            Dense.Vector.toSparseUnsorted clContext workGroupSize

        let copy = copy clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (vector: ClVector<'a>) ->
            match vector with
            | ClVector.Dense vector ->
                ClVector.Sparse
                <| toSparse processor allocationMode vector
            | ClVector.Sparse _ -> copy processor allocationMode vector

    /// <summary>
    /// Densifies the given vector if it is in a sparse format.
    /// If the given vector is already dense, copies it.
    /// </summary>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let toDense (clContext: ClContext) workGroupSize =
        let toDense =
            Sparse.Vector.toDense clContext workGroupSize

        let copy = ClArray.copy clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (vector: ClVector<'a>) ->
            match vector with
            | ClVector.Dense vector ->
                ClVector.Dense
                <| copy processor allocationMode vector vector.Length
            | ClVector.Sparse vector ->
                ClVector.Dense
                <| toDense processor allocationMode vector

    let private assignByMaskGeneral<'a, 'b when 'a: struct and 'b: struct> op (clContext: ClContext) workGroupSize =

        let sparseFillVector =
            Sparse.Vector.assignByMask op clContext workGroupSize

        let denseFillVector =
            Dense.Vector.assignByMask op clContext workGroupSize

        fun (processor: RawCommandQueue) allocationMode (vector: ClVector<'a>) (mask: ClVector<'b>) (value: 'a) ->
            match vector, mask with
            | ClVector.Sparse vector, ClVector.Sparse mask ->
                ClVector.Sparse
                <| sparseFillVector processor allocationMode vector mask value
            | ClVector.Dense vector, ClVector.Dense mask ->
                ClVector.Dense
                <| denseFillVector processor allocationMode vector mask value
            | _ -> failwith "Vector formats are not matching."

    /// <summary>
    /// Assign given value to all entries covered by mask.
    /// </summary>
    /// <param name="op"></param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let assignByMask<'a, 'b when 'a: struct and 'b: struct> op clContext workGroupSize =
        assignByMaskGeneral<'a, 'b> (Convert.assignToOption op) clContext workGroupSize

    /// <summary>
    /// Assign given value to all entries NOT covered by mask.
    /// </summary>
    /// <param name="op"></param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let assignByMaskComplemented<'a, 'b when 'a: struct and 'b: struct> op clContext workGroupSize =
        assignByMaskGeneral<'a, 'b> (Convert.assignComplementedToOption op) clContext workGroupSize

    let private assignByMaskInPlaceGeneral<'a, 'b when 'a: struct and 'b: struct>
        op
        (clContext: ClContext)
        workGroupSize
        =

        let assignByDense =
            Dense.Vector.assignByMaskInPlace op clContext workGroupSize

        let assignBySparse =
            Dense.Vector.assignBySparseMaskInPlace op clContext workGroupSize

        fun (processor: RawCommandQueue) (vector: ClVector<'a>) (mask: ClVector<'b>) (value: 'a) ->
            match vector, mask with
            | ClVector.Dense vector, ClVector.Dense mask -> assignByDense processor vector mask value vector
            | ClVector.Dense vector, ClVector.Sparse mask -> assignBySparse processor vector mask value vector
            | _ -> failwith "Unsupported format"

    /// <summary>
    /// Assign given value to all entries covered by mask.
    /// Does it in-place.
    /// </summary>
    /// <param name="op"></param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let assignByMaskInPlace<'a, 'b when 'a: struct and 'b: struct> op clContext workGroupSize =
        assignByMaskInPlaceGeneral<'a, 'b> (Convert.assignToOption op) clContext workGroupSize

    /// <summary>
    /// Applying the given function to the corresponding elements of the two given arrays pairwise.
    /// Stores the result in the left vector.
    /// </summary>
    /// <remarks>
    /// The two input arrays must have the same lengths.
    /// </remarks>
    /// <param name="map">The function to transform the pairs of the input elements.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let map2InPlace (map: Expr<'a option -> 'b option -> 'a option>) (clContext: ClContext) workGroupSize =
        let map2Dense =
            Dense.Vector.map2InPlace map clContext workGroupSize

        fun (processor: RawCommandQueue) (leftVector: ClVector<'a>) (rightVector: ClVector<'b>) ->
            match leftVector, rightVector with
            | ClVector.Dense left, ClVector.Dense right -> map2Dense processor left right left
            | _ -> failwith "Unsupported vector format"

    /// <summary>
    /// Applying the given function to the corresponding elements of the two given arrays pairwise.
    /// Stores the result in the given vector.
    /// </summary>
    /// <remarks>
    /// The two input arrays must have the same lengths.
    /// </remarks>
    /// <param name="map">The function to transform the pairs of the input elements.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let map2To (map: Expr<'a option -> 'b option -> 'c option>) (clContext: ClContext) workGroupSize =
        let map2Dense =
            Dense.Vector.map2InPlace map clContext workGroupSize

        fun (processor: RawCommandQueue) (leftVector: ClVector<'a>) (rightVector: ClVector<'b>) (resultVector: ClVector<'c>) ->
            match leftVector, rightVector, resultVector with
            | ClVector.Dense left, ClVector.Dense right, ClVector.Dense result -> map2Dense processor left right result
            | _ -> failwith "Unsupported vector format"

    /// <summary>
    /// Applying the given function to the corresponding elements of the two given arrays pairwise.
    /// Returns new vector.
    /// </summary>
    /// <remarks>
    /// The two input arrays must have the same lengths.
    /// </remarks>
    /// <param name="map">The function to transform the pairs of the input elements.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let map2Dense (map: Expr<'a option -> 'b option -> 'a option>) (clContext: ClContext) workGroupSize =
        let map2Dense =
            Dense.Vector.map2 map clContext workGroupSize

        fun (processor: RawCommandQueue) allocationFlag (leftVector: ClVector<'a>) (rightVector: ClVector<'b>) ->
            match leftVector, rightVector with
            | ClVector.Dense left, ClVector.Dense right -> map2Dense processor allocationFlag left right
            | _ -> failwith "Unsupported vector format"

    /// <summary>
    /// Applying the given function to the corresponding elements of the two given arrays pairwise.
    /// Returns new vector as option.
    /// </summary>
    /// <remarks>
    /// The two input arrays must have the same lengths.
    /// </remarks>
    /// <param name="map">The function to transform the pairs of the input elements.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let map2Sparse (map: Expr<'a option -> 'b option -> 'a option>) (clContext: ClContext) workGroupSize =
        let map2Sparse =
            Sparse.Map2.run map clContext workGroupSize

        let map2SparseDense =
            Sparse.Map2.runSparseDense map clContext workGroupSize

        fun (processor: RawCommandQueue) allocationFlag (leftVector: ClVector<'a>) (rightVector: ClVector<'b>) ->
            match leftVector, rightVector with
            | ClVector.Sparse left, ClVector.Sparse right ->
                Option.map ClVector.Sparse (map2Sparse processor allocationFlag left right)
            | ClVector.Sparse left, ClVector.Dense right ->
                Option.map ClVector.Sparse (map2SparseDense processor allocationFlag left right)
            | _ -> failwith "Unsupported vector format"

    /// <summary>
    /// Check if vector contains such element that satisfies the predicate.
    /// </summary>
    /// <param name="predicate"></param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let exists (predicate: Expr<'a option -> bool>) (clContext: ClContext) workGroupSize =

        let existsDense =
            ClArray.exists predicate clContext workGroupSize

        fun (processor: RawCommandQueue) (vector: ClVector<'a>) ->
            match vector with
            | ClVector.Dense vector -> existsDense processor vector
            | _ -> failwith "Unsupported format"

    /// <summary>
    /// Applies a function to each value of the vector, threading an accumulator argument through the computation.
    /// Begin by applying the function to the first two values.
    /// Then feed this result into the function along with the third value and so on.
    /// Return the final result.
    /// </summary>
    /// <remarks>
    /// Implicit zeroes are ignored during the computation.
    /// </remarks>
    /// <param name="opAdd"></param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let reduce (opAdd: Expr<'a -> 'a -> 'a>) (clContext: ClContext) workGroupSize =
        let sparseReduce =
            Sparse.Vector.reduce opAdd clContext workGroupSize

        let denseReduce =
            Dense.Vector.reduce opAdd clContext workGroupSize

        fun (processor: RawCommandQueue) (vector: ClVector<'a>) ->
            match vector with
            | ClVector.Sparse vector -> sparseReduce processor vector
            | ClVector.Dense vector -> denseReduce processor vector
