namespace GraphBLAS.FSharp.Backend

open Brahma.FSharp.OpenCL

module internal ScatterRowPointers =
    /// <summary>
    /// Changes array of row pointers so that if the scattering was applied to array of column indices or values.
    /// </summary>
    ///<param name="clContext">.</param>
    ///<param name="workGroupSize">.</param>
    ///<param name="processor">.</param>
    ///<param name="positions">
    /// Indices of the elements in the array that would be the result of scattering.
    /// The very first index must be zero.
    /// Every index must be the same as the previous one or more by one.
    /// </param>
    ///<param name="initialLength">Length of the array that would be the input array for scattering.</param>
    ///<param name="length">Length of the array that would be the result of scattering.</param>
    ///<param name="rowPointers">.</param>
    let runInPlace
        (clContext: ClContext)
        workGroupSize
        (processor: MailboxProcessor<_>)
        (positions: ClArray<int>)
        (initialLength: int)
        (length: int)
        (rowPointers: ClArray<int>) =

        let rowPointersLength = rowPointers.Length

        let setPositions =
            <@
                fun (ndRange: Range1D)
                    (rowPointers: ClArray<int>)
                    (positions: ClArray<int>)
                    (initialLength: int)
                    (length: int) ->

                    let i = ndRange.GlobalID0
                    if i < rowPointersLength then
                        let buff = rowPointers.[i]
                        if buff = initialLength then
                            rowPointers.[i] <- length
                        else
                            rowPointers.[i] <- positions.[buff]
            @>

        let kernel = clContext.CreateClKernel(setPositions)
        let ndRange = Range1D.CreateValid(rowPointersLength, workGroupSize)
        processor.Post(
            Msg.MsgSetArguments
                (fun () ->
                    kernel.ArgumentsSetter
                        ndRange
                        rowPointers
                        positions
                        initialLength
                        length)
        )
        processor.Post(Msg.CreateRunMsg<_, _>(kernel))
