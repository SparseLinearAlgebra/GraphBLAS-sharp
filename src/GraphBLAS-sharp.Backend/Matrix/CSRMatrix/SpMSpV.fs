namespace GraphBLAS.FSharp.Backend.CSRMatrix

open Brahma.FSharp.OpenCL
open GraphBLAS.FSharp
open GraphBLAS.FSharp.Backend.Common

module internal rec SpMSpV =
    let unmasked (matrix: CSRMatrix<'a>) (vector: COOVector<'a>) (semiring: ISemiring<'a>) =
        opencl {
            if matrix.Values.Length = 0
               || vector.Values.Length = 0 then
                return
                    { Size = matrix.RowCount
                      Indices = [||]
                      Values = [||] }

            else
                let rowCount = matrix.RowCount
                let vectorNnz = vector.Values.Length
                let wgSize = Utils.defaultWorkGroupSize

                let (ClosedBinaryOp plus) = semiring.Plus
                let (ClosedBinaryOp times) = semiring.Times
                let zero = semiring.Zero

                let calcValuesPerRow =
                    <@ fun (range: Range1D) (matrixRowPointers: int []) (matrixColumnIndices: int []) (matrixValues: 'a []) (vectorIndices: int []) (vectorValues: 'a []) (countOfProductsPerRow: int []) (valuesPerRow: 'a []) ->

                        let gid = range.GlobalID0
                        let lid = range.LocalID0
                        let groupId = gid / wgSize // rowId

                        let localCountAccum = localArray<int> wgSize
                        localCountAccum.[lid] <- 0

                        let localValuesAccum = localArray<'a> wgSize
                        localValuesAccum.[lid] <- zero

                        barrier ()

                        let mutable i = matrixRowPointers.[groupId] + lid
                        let _end = matrixRowPointers.[groupId + 1]

                        while i < _end do
                            let col = matrixColumnIndices.[i]
                            let value = matrixValues.[i]

                            let mutable l = 0
                            let mutable r = vectorNnz
                            let mutable m = l + ((r - l) / 2)
                            let mutable idx = -1
                            let mutable _break = false

                            while l < r && not _break do
                                if vectorIndices.[m] = col then
                                    idx <- m
                                    _break <- true
                                elif vectorIndices.[m] < col then
                                    l <- m + 1
                                else
                                    r <- m

                                m <- l + ((r - l) / 2)

                            if idx <> -1 then
                                let vectorValue = vectorValues.[idx]
                                localCountAccum.[lid] <- localCountAccum.[lid] + 1
                                localValuesAccum.[lid] <- (%plus) localValuesAccum.[lid] ((%times) value vectorValue)

                            i <- i + wgSize

                        barrier ()

                        if lid = 0 then
                            let mutable countAcc = 0
                            let mutable valueAcc = zero

                            for i = 0 to wgSize - 1 do
                                countAcc <- countAcc + localCountAccum.[i]
                                valueAcc <- (%plus) valueAcc localValuesAccum.[i]

                            countOfProductsPerRow.[groupId] <- countAcc
                            valuesPerRow.[groupId] <- valueAcc @>

                let countOfProductsPerRow = Array.zeroCreate<int> rowCount
                let valuesPerRow = Array.zeroCreate<'a> rowCount

                do!
                    runCommand calcValuesPerRow
                    <| fun kernelPrepare ->
                        kernelPrepare
                        <| Range1D(rowCount * Utils.defaultWorkGroupSize, Utils.defaultWorkGroupSize)
                        <| matrix.RowPointers
                        <| matrix.ColumnIndices
                        <| matrix.Values
                        <| vector.Indices
                        <| vector.Values
                        <| countOfProductsPerRow
                        <| valuesPerRow

                let getNonzeroBitmap =
                    <@ fun (range: Range1D) (count: int []) (bitmap: int []) ->

                        let gid = range.GlobalID0

                        if gid < rowCount && count.[gid] = 0 then
                            bitmap.[gid] <- 0 @>

                let bitmap = Array.create<int> rowCount 1

                do!
                    runCommand getNonzeroBitmap
                    <| fun kernelPrepare ->
                        kernelPrepare
                        <| Range1D(Utils.getDefaultGlobalSize rowCount, Utils.defaultWorkGroupSize)
                        <| countOfProductsPerRow
                        <| bitmap

                let! (positions, totalSum) = PrefixSum.runExclude bitmap
                failwith "FIX ME! And rewrite."
                //let! _ = ToHost totalSum
                let resultLength = totalSum.[0]

                if resultLength = 0 then
                    return
                        { Size = matrix.RowCount
                          Indices = [||]
                          Values = [||] }

                else
                    let getOutputVector =
                        <@ fun (range: Range1D) (count: int []) (values: 'a []) (positions: int []) (outputValues: 'a []) (outputIndices: int []) ->

                            let gid = range.GlobalID0

                            if gid < rowCount && count.[gid] <> 0 then
                                outputValues.[positions.[gid]] <- values.[gid]
                                outputIndices.[positions.[gid]] <- gid @>

                    let outputValues = Array.zeroCreate<'a> resultLength
                    let outputIndices = Array.zeroCreate<int> resultLength

                    do!
                        runCommand getOutputVector
                        <| fun kernelPrepare ->
                            kernelPrepare
                            <| Range1D(Utils.getDefaultGlobalSize rowCount, Utils.defaultWorkGroupSize)
                            <| countOfProductsPerRow
                            <| valuesPerRow
                            <| positions
                            <| outputValues
                            <| outputIndices

                    return
                        { Size = rowCount
                          Indices = outputIndices
                          Values = outputValues }
        }

    let masked (matrix: CSRMatrix<'a>) (vector: COOVector<'a>) (semiring: ISemiring<'a>) (mask: Mask1D) =
        opencl {
            if matrix.Values.Length = 0
               || vector.Values.Length = 0 then
                return
                    { Size = matrix.RowCount
                      Indices = [||]
                      Values = [||] }

            elif mask.Indices.Length = 0 && not mask.IsComplemented
                 || mask.Indices.Length = mask.Size
                    && mask.IsComplemented then
                return
                    { Size = matrix.RowCount
                      Indices = [||]
                      Values = [||] }

            else
                let rowCount = matrix.RowCount
                let vectorNnz = vector.Values.Length
                let wgSize = Utils.defaultWorkGroupSize
                let maskNnz = mask.Indices.Length

                let (ClosedBinaryOp plus) = semiring.Plus
                let (ClosedBinaryOp times) = semiring.Times
                let zero = semiring.Zero

                let calcValuesPerRow =
                    <@ fun (range: Range1D) (mask: int []) (matrixRowPointers: int []) (matrixColumnIndices: int []) (matrixValues: 'a []) (vectorIndices: int []) (vectorValues: 'a []) (countOfProductsPerRow: int []) (valuesPerRow: 'a []) ->

                        let gid = range.GlobalID0
                        let lid = range.LocalID0
                        let groupId = gid / wgSize
                        let rowId = mask.[groupId]

                        let localCountAccum = localArray<int> wgSize
                        localCountAccum.[lid] <- 0

                        let localValuesAccum = localArray<'a> wgSize
                        localValuesAccum.[lid] <- zero

                        barrier ()

                        let mutable i = matrixRowPointers.[rowId] + lid
                        let _end = matrixRowPointers.[rowId + 1]

                        while i < _end do
                            let col = matrixColumnIndices.[i]
                            let value = matrixValues.[i]

                            let mutable l = 0
                            let mutable r = vectorNnz
                            let mutable m = l + ((r - l) / 2)
                            let mutable idx = -1
                            let mutable _break = false

                            while l < r && not _break do
                                if vectorIndices.[m] = col then
                                    idx <- m
                                    _break <- true
                                elif vectorIndices.[m] < col then
                                    l <- m + 1
                                else
                                    r <- m

                                m <- l + ((r - l) / 2)

                            if idx <> -1 then
                                let vectorValue = vectorValues.[idx]
                                localCountAccum.[lid] <- localCountAccum.[lid] + 1
                                localValuesAccum.[lid] <- (%plus) localValuesAccum.[lid] ((%times) value vectorValue)

                            i <- i + wgSize

                        barrier ()

                        if lid = 0 then
                            let mutable countAcc = 0
                            let mutable valueAcc = zero

                            for i = 0 to wgSize - 1 do
                                countAcc <- countAcc + localCountAccum.[i]
                                valueAcc <- (%plus) valueAcc localValuesAccum.[i]

                            countOfProductsPerRow.[rowId] <- countAcc
                            valuesPerRow.[rowId] <- valueAcc @>

                let countOfProductsPerRow = Array.zeroCreate<int> rowCount
                let valuesPerRow = Array.zeroCreate<'a> rowCount

                do!
                    runCommand calcValuesPerRow
                    <| fun kernelPrepare ->
                        kernelPrepare
                        <| Range1D(maskNnz * Utils.defaultWorkGroupSize, Utils.defaultWorkGroupSize)
                        <| mask.Indices
                        <| matrix.RowPointers
                        <| matrix.ColumnIndices
                        <| matrix.Values
                        <| vector.Indices
                        <| vector.Values
                        <| countOfProductsPerRow
                        <| valuesPerRow

                let getNonzeroBitmap =
                    <@ fun (range: Range1D) (count: int []) (bitmap: int []) ->

                        let gid = range.GlobalID0

                        if gid < rowCount && count.[gid] = 0 then
                            bitmap.[gid] <- 0 @>

                let bitmap = Array.create<int> rowCount 1

                do!
                    runCommand getNonzeroBitmap
                    <| fun kernelPrepare ->
                        kernelPrepare
                        <| Range1D(Utils.getDefaultGlobalSize rowCount, Utils.defaultWorkGroupSize)
                        <| countOfProductsPerRow
                        <| bitmap

                let! (positions, totalSum) = PrefixSum.runExclude bitmap
                failwith "FIX ME! And rewrite."
                //let! _ = ToHost totalSum
                let resultLength = totalSum.[0]

                if resultLength = 0 then
                    return
                        { Size = matrix.RowCount
                          Indices = [||]
                          Values = [||] }

                else
                    let getOutputVector =
                        <@ fun (range: Range1D) (count: int []) (values: 'a []) (positions: int []) (outputValues: 'a []) (outputIndices: int []) ->

                            let gid = range.GlobalID0

                            if gid < rowCount && count.[gid] <> 0 then
                                outputValues.[positions.[gid]] <- values.[gid]
                                outputIndices.[positions.[gid]] <- gid @>

                    let outputValues = Array.zeroCreate<'a> resultLength
                    let outputIndices = Array.zeroCreate<int> resultLength

                    do!
                        runCommand getOutputVector
                        <| fun kernelPrepare ->
                            kernelPrepare
                            <| Range1D(Utils.getDefaultGlobalSize rowCount, Utils.defaultWorkGroupSize)
                            <| countOfProductsPerRow
                            <| valuesPerRow
                            <| positions
                            <| outputValues
                            <| outputIndices

                    return
                        { Size = rowCount
                          Indices = outputIndices
                          Values = outputValues }
        }
