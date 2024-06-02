namespace GraphBLAS.FSharp

open Brahma.FSharp
open Microsoft.FSharp.Core
open Microsoft.FSharp.Quotations
open GraphBLAS.FSharp.Objects
open GraphBLAS.FSharp.Objects.ClContextExtensions
open GraphBLAS.FSharp.Backend.Matrix
open GraphBLAS.FSharp.Backend.Vector
open GraphBLAS.FSharp.Backend.Operations

[<RequireQualifiedAccess>]
module Operations =
    module Vector =
        /// <summary>
        /// Builds a new vector whose elements are the results of applying the given function
        /// to each of the elements of the vector.
        /// </summary>
        /// <param name="op">
        /// A function to transform values of the input vector.
        /// Operand and result types should be optional to distinguish explicit and implicit zeroes.
        /// </param>
        /// <param name="clContext">OpenCL context.</param>
        /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
        let map (op: Expr<'a option -> 'b option>) (clContext: ClContext) workGroupSize =
            let mapSparse =
                Sparse.Vector.map op clContext workGroupSize

            let mapDense =
                Dense.Vector.map op clContext workGroupSize

            fun (processor: RawCommandQueue) allocationMode matrix ->
                match matrix with
                | ClVector.Sparse v ->
                    mapSparse processor allocationMode v
                    |> ClVector.Sparse
                | ClVector.Dense v ->
                    mapDense processor allocationMode v
                    |> ClVector.Dense

        /// <summary>
        /// Builds a new vector whose values are the results of applying the given function
        /// to the corresponding pairs of values from the two vectors.
        /// </summary>
        /// <param name="op">
        /// A function to transform pairs of values from the input vectors.
        /// Operands and result types should be optional to distinguish explicit and implicit zeroes.
        /// </param>
        /// <remarks>
        /// Formats of the given vectors should match, otherwise an exception will be thrown.
        /// </remarks>
        /// <param name="clContext">OpenCL context.</param>
        /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
        let map2 (op: Expr<'a option -> 'b option -> 'c option>) (clContext: ClContext) workGroupSize =
            let map2Dense =
                Dense.Vector.map2 op clContext workGroupSize

            let map2Sparse =
                Sparse.Vector.map2 op clContext workGroupSize

            fun (processor: RawCommandQueue) allocationMode (leftVector: ClVector<'a>) (rightVector: ClVector<'b>) ->
                match leftVector, rightVector with
                | ClVector.Dense left, ClVector.Dense right ->
                    map2Dense processor allocationMode left right
                    |> ClVector.Dense
                    |> Some
                | ClVector.Sparse left, ClVector.Sparse right ->
                    Option.map ClVector.Sparse (map2Sparse processor allocationMode left right)
                | _ -> failwith "Vector formats are not matching."

        /// <summary>
        /// Builds a new vector whose values are the results of applying the given function
        /// to the corresponding pairs of values from the two vectors.
        /// </summary>
        /// <param name="op">
        /// A function to transform pairs of values from the input vectors.
        /// Operation assumption: one of the operands should always be non-zero.
        /// </param>
        /// <remarks>
        /// Formats of the given vectors should match, otherwise an exception will be thrown.
        /// </remarks>
        /// <param name="clContext">OpenCL context.</param>
        /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
        let map2AtLeastOne (op: Expr<AtLeastOne<'a, 'b> -> 'c option>) (clContext: ClContext) workGroupSize =
            let map2Sparse =
                Sparse.Vector.map2AtLeastOne op clContext workGroupSize

            let map2Dense =
                Dense.Vector.map2AtLeastOne op clContext workGroupSize

            fun (processor: RawCommandQueue) allocationMode (leftVector: ClVector<'a>) (rightVector: ClVector<'b>) ->
                match leftVector, rightVector with
                | ClVector.Sparse left, ClVector.Sparse right ->
                    Option.map ClVector.Sparse (map2Sparse processor allocationMode left right)
                | ClVector.Dense left, ClVector.Dense right ->
                    map2Dense processor allocationMode left right
                    |> ClVector.Dense
                    |> Some
                | _ -> failwith "Vector formats are not matching."

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
                | ClVector.Dense left, ClVector.Dense right, ClVector.Dense result ->
                    map2Dense processor left right result
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

    module Matrix =
        /// <summary>
        /// Builds a new matrix whose elements are the results of applying the given function
        /// to each of the elements of the matrix.
        /// </summary>
        /// <param name="op">
        /// A function to transform values of the input matrix.
        /// Operand and result types should be optional to distinguish explicit and implicit zeroes
        /// </param>
        /// <param name="clContext">OpenCL context.</param>
        /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
        let map (op: Expr<'a option -> 'b option>) (clContext: ClContext) workGroupSize =
            let mapCOO =
                COO.Matrix.map op clContext workGroupSize

            let mapCSR =
                CSR.Matrix.map op clContext workGroupSize

            let transposeCOO =
                COO.Matrix.transposeInPlace clContext workGroupSize

            fun (processor: RawCommandQueue) allocationMode matrix ->
                match matrix with
                | ClMatrix.COO m -> mapCOO processor allocationMode m |> ClMatrix.COO
                | ClMatrix.CSR m -> mapCSR processor allocationMode m |> ClMatrix.COO
                | ClMatrix.CSC m ->
                    (mapCSR processor allocationMode m.ToCSR)
                    |> transposeCOO processor
                    |> ClMatrix.COO
                | _ -> failwith "Not yet implemented"

        /// <summary>
        /// Builds a new matrix whose values are the results of applying the given function
        /// to the corresponding pairs of values from the two matrices.
        /// </summary>
        /// <param name="op">
        /// A function to transform pairs of values from the input matrices.
        /// Operands and result types should be optional to distinguish explicit and implicit zeroes
        /// </param>
        /// <remarks>
        /// Formats of the given matrices should match, otherwise an exception will be thrown.
        /// </remarks>
        /// <param name="clContext">OpenCL context.</param>
        /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
        let map2 (op: Expr<'a option -> 'b option -> 'c option>) (clContext: ClContext) workGroupSize =
            let map2COO =
                COO.Matrix.map2 op clContext workGroupSize

            let map2CSR =
                CSR.Matrix.map2 op clContext workGroupSize

            let transposeCOO =
                COO.Matrix.transposeInPlace clContext workGroupSize

            fun (processor: RawCommandQueue) allocationMode matrix1 matrix2 ->
                match matrix1, matrix2 with
                | ClMatrix.COO m1, ClMatrix.COO m2 ->
                    map2COO processor allocationMode m1 m2
                    |> ClMatrix.COO
                | ClMatrix.CSR m1, ClMatrix.CSR m2 ->
                    map2CSR processor allocationMode m1 m2
                    |> ClMatrix.COO
                | ClMatrix.CSC m1, ClMatrix.CSC m2 ->
                    (map2CSR processor allocationMode m1.ToCSR m2.ToCSR)
                    |> transposeCOO processor
                    |> ClMatrix.COO
                | _ -> failwith "Matrix formats are not matching"

        /// <summary>
        /// Builds a new matrix whose values are the results of applying the given function
        /// to the corresponding pairs of values from the two matrices.
        /// </summary>
        /// <param name="op">
        /// A function to transform pairs of values from the input matrices.
        /// Operation assumption: one of the operands should always be non-zero.
        /// </param>
        /// <remarks>
        /// Formats of the given matrices should match, otherwise an exception will be thrown.
        /// </remarks>
        /// <param name="clContext">OpenCL context.</param>
        /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
        let map2AtLeastOne (op: Expr<AtLeastOne<'a, 'b> -> 'c option>) (clContext: ClContext) workGroupSize =
            let COOMap2 =
                COO.Matrix.map2AtLeastOne clContext op workGroupSize

            let CSRMap2 =
                CSR.Matrix.map2AtLeastOne clContext op workGroupSize

            let COOTranspose =
                COO.Matrix.transposeInPlace clContext workGroupSize

            fun (processor: RawCommandQueue) allocationMode matrix1 matrix2 ->
                match matrix1, matrix2 with
                | ClMatrix.COO m1, ClMatrix.COO m2 ->
                    COOMap2 processor allocationMode m1 m2
                    |> ClMatrix.COO
                | ClMatrix.CSR m1, ClMatrix.CSR m2 ->
                    CSRMap2 processor allocationMode m1 m2
                    |> ClMatrix.COO
                | ClMatrix.CSC m1, ClMatrix.CSC m2 ->
                    (CSRMap2 processor allocationMode m1.ToCSR m2.ToCSR)
                    |> COOTranspose processor
                    |> ClMatrix.COO
                | _ -> failwith "Matrix formats are not matching"

    /// <summary>
    /// Matrix-vector multiplication.
    /// Vector, that is going to be the result of the operation, should be passed as an argument.
    /// </summary>
    /// <param name="add">Type of binary function to reduce entries.</param>
    /// <param name="mul">Type of binary function to combine entries.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let SpMVInPlace
        (add: Expr<'c option -> 'c option -> 'c option>)
        (mul: Expr<'a option -> 'b option -> 'c option>)
        (clContext: ClContext)
        workGroupSize
        =

        let runTo =
            SpMV.runTo add mul clContext workGroupSize

        fun (queue: RawCommandQueue) (matrix: ClMatrix<'a>) (vector: ClVector<'b>) (result: ClVector<'c>) ->
            match matrix, vector, result with
            | ClMatrix.CSR m, ClVector.Dense v, ClVector.Dense r -> runTo queue m v r
            | _ -> failwith "Not implemented yet"

    /// <summary>
    /// CSR Matrix - dense vector multiplication.
    /// </summary>
    /// <param name="add">Type of binary function to reduce entries.</param>
    /// <param name="mul">Type of binary function to combine entries.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let SpMV
        (add: Expr<'c option -> 'c option -> 'c option>)
        (mul: Expr<'a option -> 'b option -> 'c option>)
        (clContext: ClContext)
        workGroupSize
        =

        let run = SpMV.run add mul clContext workGroupSize

        fun (queue: RawCommandQueue) allocationFlag (matrix: ClMatrix<'a>) (vector: ClVector<'b>) ->
            match matrix, vector with
            | ClMatrix.CSR m, ClVector.Dense v -> run queue allocationFlag m v |> ClVector.Dense
            | _ -> failwith "Not implemented yet"

    /// <summary>
    /// CSR Matrix - sparse vector multiplication. Optimized for bool OR and AND operations by skipping reduction stage.
    /// </summary>
    /// <param name="add">Type of binary function to reduce entries.</param>
    /// <param name="mul">Type of binary function to combine entries.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let SpMSpVBool
        (add: Expr<bool option -> bool option -> bool option>)
        (mul: Expr<bool option -> bool option -> bool option>)
        (clContext: ClContext)
        workGroupSize
        =

        let run =
            SpMSpV.runBoolStandard add mul clContext workGroupSize

        fun (queue: RawCommandQueue) (matrix: ClMatrix<bool>) (vector: ClVector<bool>) ->
            match matrix, vector with
            | ClMatrix.CSR m, ClVector.Sparse v -> Option.map ClVector.Sparse (run queue m v)
            | _ -> failwith "Not implemented yet"

    /// <summary>
    /// CSR Matrix - sparse vector multiplication with mask. Mask is complemented.
    /// </summary>
    /// <param name="add">Type of binary function to reduce entries.</param>
    /// <param name="mul">Type of binary function to combine entries.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let SpMSpVMasked
        (add: Expr<'c option -> 'c option -> 'c option>)
        (mul: Expr<'a option -> 'b option -> 'c option>)
        (clContext: ClContext)
        workGroupSize
        =

        let run =
            SpMSpV.Masked.runMasked add mul clContext workGroupSize

        fun (queue: RawCommandQueue) (matrix: ClMatrix<'a>) (vector: ClVector<'b>) (mask: ClVector<'d>) ->
            match matrix, vector, mask with
            | ClMatrix.CSR m, ClVector.Sparse v, ClVector.Dense mask -> Option.map ClVector.Sparse (run queue m v mask)
            | _ -> failwith "Not implemented yet"

    /// <summary>
    /// CSR Matrix - sparse vector multiplication with mask. Mask is complemented. Optimized for bool OR and AND operations by skipping reduction stage.
    /// </summary>
    /// <param name="add">Type of binary function to reduce entries.</param>
    /// <param name="mul">Type of binary function to combine entries.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let SpMSpVMaskedBool
        (add: Expr<bool option -> bool option -> bool option>)
        (mul: Expr<bool option -> bool option -> bool option>)
        (clContext: ClContext)
        workGroupSize
        =

        let run =
            SpMSpV.Masked.runMaskedBoolStandard add mul clContext workGroupSize

        fun (queue: RawCommandQueue) (matrix: ClMatrix<'a>) (vector: ClVector<'b>) (mask: ClVector<'d>) ->
            match matrix, vector, mask with
            | ClMatrix.CSR m, ClVector.Sparse v, ClVector.Dense mask -> Option.map ClVector.Sparse (run queue m v mask)
            | _ -> failwith "Not implemented yet"

    /// <summary>
    /// CSR Matrix - sparse vector multiplication.
    /// </summary>
    /// <param name="add">Type of binary function to reduce entries.</param>
    /// <param name="mul">Type of binary function to combine entries.</param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let SpMSpV
        (add: Expr<'c option -> 'c option -> 'c option>)
        (mul: Expr<'a option -> 'b option -> 'c option>)
        (clContext: ClContext)
        workGroupSize
        =

        let run =
            SpMSpV.run add mul clContext workGroupSize

        fun (queue: RawCommandQueue) (matrix: ClMatrix<'a>) (vector: ClVector<'b>) ->
            match matrix, vector with
            | ClMatrix.CSR m, ClVector.Sparse v -> Option.map ClVector.Sparse (run queue m v)
            | _ -> failwith "Not implemented yet"

    /// <summary>
    /// Kronecker product for matrices.
    /// </summary>
    /// <param name="op">
    /// Element-wise operation.
    /// Operands and result types should be optional to distinguish explicit and implicit zeroes
    /// </param>
    /// <param name="clContext">OpenCL context.</param>
    /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
    let kronecker (op: Expr<'a option -> 'b option -> 'c option>) (clContext: ClContext) workGroupSize =
        let run = Kronecker.run clContext workGroupSize op

        fun (queue: RawCommandQueue) allocationFlag (matrix1: ClMatrix<'a>) (matrix2: ClMatrix<'b>) ->
            match matrix1, matrix2 with
            | ClMatrix.CSR m1, ClMatrix.CSR m2 ->
                let result = run queue allocationFlag m1 m2
                Option.map ClMatrix.COO result
            | _ -> failwith "Both matrices should be in CSR format."

    /// <summary>
    /// Sparse general matrix-matrix multiplication.
    /// </summary>
    module SpGeMM =
        /// <summary>
        /// Masked matrix-matrix multiplication.
        /// </summary>
        /// <param name="opAdd">Type of binary function to reduce entries.</param>
        /// <param name="opMul">Type of binary function to combine entries.</param>
        /// <param name="clContext">OpenCL context.</param>
        /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
        let masked
            (opAdd: Expr<'c -> 'c -> 'c option>)
            (opMul: Expr<'a -> 'b -> 'c option>)
            (clContext: ClContext)
            workGroupSize
            =

            let runCSRnCSC =
                SpGeMM.Masked.run opAdd opMul clContext workGroupSize

            fun (queue: RawCommandQueue) (matrix1: ClMatrix<'a>) (matrix2: ClMatrix<'b>) (mask: ClMatrix<_>) ->
                match matrix1, matrix2, mask with
                | ClMatrix.CSR m1, ClMatrix.CSC m2, ClMatrix.COO mask -> runCSRnCSC queue m1 m2 mask |> ClMatrix.COO
                | _ -> failwith "Matrix formats are not matching"

        /// <summary>
        /// Generalized matrix-matrix multiplication.
        /// </summary>
        /// <param name="opAdd">Type of binary function to reduce entries.</param>
        /// <param name="opMul">Type of binary function to combine entries.</param>
        /// <param name="clContext">OpenCL context.</param>
        /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
        let expand
            (opAdd: Expr<'c -> 'c -> 'c option>)
            (opMul: Expr<'a -> 'b -> 'c option>)
            (clContext: ClContext)
            workGroupSize
            =

            let run =
                SpGeMM.Expand.run opAdd opMul clContext workGroupSize

            fun (processor: RawCommandQueue) allocationMode (leftMatrix: ClMatrix<'a>) (rightMatrix: ClMatrix<'b>) ->
                match leftMatrix, rightMatrix with
                | ClMatrix.CSR leftMatrix, ClMatrix.CSR rightMatrix ->
                    let allocCapacity =
                        List.max [ sizeof<'a>
                                   sizeof<'c>
                                   sizeof<'b> ]
                        |> uint64
                        |> (*) 1UL<Byte>

                    let resultCapacity =
                        (clContext.MaxMemAllocSize / allocCapacity) / 3UL

                    let resultCapacity =
                        (min
                         <| uint64 System.Int32.MaxValue
                         <| resultCapacity)
                        |> int

                    run processor allocationMode resultCapacity leftMatrix rightMatrix
                | _ -> failwith "Matrix formats are not matching"

        module COO =
            /// <summary>
            /// Generalized matrix-matrix multiplication. Left matrix should be in COO format.
            /// </summary>
            /// <param name="opAdd">Type of binary function to reduce entries.</param>
            /// <param name="opMul">Type of binary function to combine entries.</param>
            /// <param name="clContext">OpenCL context.</param>
            /// <param name="workGroupSize">Should be a power of 2 and greater than 1.</param>
            let expand
                (opAdd: Expr<'c -> 'c -> 'c option>)
                (opMul: Expr<'a -> 'b -> 'c option>)
                (clContext: ClContext)
                workGroupSize
                =

                let run =
                    SpGeMM.Expand.COO.run opAdd opMul clContext workGroupSize

                fun (processor: RawCommandQueue) allocationMode (leftMatrix: ClMatrix<'a>) (rightMatrix: ClMatrix<'b>) ->
                    match leftMatrix, rightMatrix with
                    | ClMatrix.COO leftMatrix, ClMatrix.CSR rightMatrix ->
                        let allocCapacity =
                            List.max [ sizeof<'a>
                                       sizeof<'c>
                                       sizeof<'b> ]
                            |> uint64
                            |> (*) 1UL<Byte>

                        let resultCapacity =
                            (clContext.MaxMemAllocSize / allocCapacity) / 3UL

                        let resultCapacity =
                            (min
                             <| uint64 System.Int32.MaxValue
                             <| resultCapacity)
                            |> int

                        run processor allocationMode resultCapacity leftMatrix rightMatrix
                    | _ -> failwith "Matrix formats are not matching"
