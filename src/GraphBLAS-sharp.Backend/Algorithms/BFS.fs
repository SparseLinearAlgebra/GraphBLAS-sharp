namespace GraphBLAS.FSharp.Backend.Algorithms

open Brahma.FSharp
open FSharp.Quotations
open GraphBLAS.FSharp
open GraphBLAS.FSharp.Objects
open GraphBLAS.FSharp.Backend.Quotes
open GraphBLAS.FSharp.Objects.ArraysExtensions
open GraphBLAS.FSharp.Objects.ClContextExtensions
open GraphBLAS.FSharp.Objects.ClCellExtensions

module internal BFS =
    let singleSource
        (add: Expr<bool option -> bool option -> bool option>)
        (mul: Expr<bool option -> bool option -> bool option>)
        (clContext: ClContext)
        workGroupSize
        =

        let spMVInPlace =
            Operations.SpMVInPlace add mul clContext workGroupSize

        let zeroCreate =
            Vector.zeroCreate clContext workGroupSize

        let ofList = Vector.ofList clContext workGroupSize

        let maskComplementedInPlace =
            Vector.map2InPlace Mask.complementedOp clContext workGroupSize

        let fillSubVectorTo =
            Vector.assignByMaskInPlace Mask.assign clContext workGroupSize

        let containsNonZero =
            Vector.exists Predicates.isSome clContext workGroupSize

        fun (queue: RawCommandQueue) (matrix: ClMatrix<bool>) (source: int) ->
            let vertexCount = matrix.RowCount

            let levels =
                zeroCreate queue DeviceOnly vertexCount Dense

            let front =
                ofList queue DeviceOnly Dense vertexCount [ source, true ]

            let mutable level = 0
            let mutable stop = false

            while not stop do
                level <- level + 1

                //Assigning new level values
                fillSubVectorTo queue levels front level

                //Getting new frontier
                spMVInPlace queue matrix front front

                maskComplementedInPlace queue front levels

                //Checking if front is empty
                stop <-
                    not
                    <| (containsNonZero queue front).ToHostAndFree queue

            front.Dispose()

            levels

    let singleSourceSparse
        (add: Expr<bool option -> bool option -> bool option>)
        (mul: Expr<bool option -> bool option -> bool option>)
        (clContext: ClContext)
        workGroupSize
        =

        let spMSpV =
            Operations.SpMSpVBool add mul clContext workGroupSize

        let zeroCreate =
            Vector.zeroCreate clContext workGroupSize

        let ofList = Vector.ofList clContext workGroupSize

        let maskComplemented =
            Vector.map2Sparse Mask.complementedOp clContext workGroupSize

        let fillSubVectorTo =
            Vector.assignByMaskInPlace Mask.assign clContext workGroupSize

        fun (queue: RawCommandQueue) (matrix: ClMatrix<bool>) (source: int) ->
            let vertexCount = matrix.RowCount

            let levels =
                zeroCreate queue DeviceOnly vertexCount Dense

            let mutable front =
                ofList queue DeviceOnly Sparse vertexCount [ source, true ]

            let mutable level = 0
            let mutable stop = false

            while not stop do
                level <- level + 1

                //Assigning new level values
                fillSubVectorTo queue levels front level

                //Getting new frontier
                match spMSpV queue matrix front with
                | None ->
                    front.Dispose()
                    stop <- true
                | Some newFrontier ->
                    front.Dispose()
                    //Filtering visited vertices
                    match maskComplemented queue DeviceOnly newFrontier levels with
                    | None ->
                        stop <- true
                        newFrontier.Dispose()
                    | Some f ->
                        front <- f
                        newFrontier.Dispose()

            levels


    let singleSourcePushPull
        (add: Expr<bool option -> bool option -> bool option>)
        (mul: Expr<bool option -> bool option -> bool option>)
        (clContext: ClContext)
        workGroupSize
        =

        let spMVInPlace =
            Operations.SpMVInPlace add mul clContext workGroupSize

        let spMSpV =
            Operations.SpMSpVMaskedBool add mul clContext workGroupSize

        let zeroCreate =
            Vector.zeroCreate clContext workGroupSize

        let ofList = Vector.ofList clContext workGroupSize

        let maskComplementedInPlace =
            Vector.map2InPlace Mask.complementedOp clContext workGroupSize

        let fillSubVectorInPlace =
            Vector.assignByMaskInPlace (Mask.assign) clContext workGroupSize

        let toSparse =
            Vector.toSparseUnsorted clContext workGroupSize

        let toDense = Vector.toDense clContext workGroupSize

        let countNNZ =
            ClArray.count Predicates.isSome clContext workGroupSize

        //Push or pull functions
        let getNNZ (queue: RawCommandQueue) (v: ClVector<bool>) =
            match v with
            | ClVector.Sparse v -> v.NNZ
            | ClVector.Dense v -> countNNZ queue v

        let SPARSITY = 0.05f

        let push nnz size =
            (float32 nnz) / (float32 size) <= SPARSITY

        fun (queue: RawCommandQueue) (matrix: ClMatrix<bool>) (source: int) ->
            let vertexCount = matrix.RowCount

            let levels =
                zeroCreate queue DeviceOnly vertexCount Dense

            let mutable frontier =
                ofList queue DeviceOnly Sparse vertexCount [ source, true ]

            let mutable level = 0
            let mutable stop = false

            while not stop do
                level <- level + 1

                //Assigning new level values
                fillSubVectorInPlace queue levels frontier level

                match frontier with
                | ClVector.Sparse _ ->
                    //Getting new frontier
                    match spMSpV queue matrix frontier levels with
                    | None ->
                        frontier.Dispose()
                        stop <- true
                    | Some newMaskedFrontier ->
                        frontier.Dispose()

                        //Push/pull
                        let NNZ = getNNZ queue newMaskedFrontier

                        if (push NNZ newMaskedFrontier.Size) then
                            frontier <- newMaskedFrontier
                        else
                            frontier <- toDense queue DeviceOnly newMaskedFrontier
                            newMaskedFrontier.Dispose()
                | ClVector.Dense oldFrontier ->
                    //Getting new frontier
                    spMVInPlace queue matrix frontier frontier

                    maskComplementedInPlace queue frontier levels

                    //Emptiness check
                    let NNZ = getNNZ queue frontier

                    stop <- NNZ = 0

                    //Push/pull
                    if not stop then
                        if (push NNZ frontier.Size) then
                            frontier <- toSparse queue DeviceOnly frontier
                            oldFrontier.Free()
                    else
                        frontier.Dispose()

            levels
