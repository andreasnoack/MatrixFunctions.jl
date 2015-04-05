module MatrixFunctions

    import Base: getindex, size
    import Base.LinAlg: expm, sqrtm

    immutable Matrix2x2{T} <: AbstractMatrix{T}
        v11::T
        v21::T
        v12::T
        v22::T
    end

    size(A::Matrix2x2) = (2,2)

    getindex(A::Matrix2x2, i::Integer) = i == 1 ? A.v11 : (i == 2 ? A.v21 : (i == 3 ? A.v12 : (i == 4 ? A.v22 : throw(BoundsError()))))
    getindex(A::Matrix2x2, i::Integer, j::Integer) = i == 1 ? (j == 1 ? A.v11 : (j == 2 ? A.v12 : throw(BoundsError))) : (i == 2 ? (j == 1 ? A.v21 : (j == 2 ? A.v22 : throw(BoundsError))) : throw(BoundsError()))

    function matmap!(F::UpperTriangular, f, T::UpperTriangular)
        n = size(T, 1)
        n == size(F, 1) || throw(DimenstionMismatch("matrices must have same size"))

        for i = 1:n
            F.data[i,i] = f(T[i,i])
        end
        for dd = 1:n - 1
            for i = 1:n - dd
                j = i + dd
                Fij = T[i, j]*(F[i, i] - F[j, j])/(T[i, i] - T[j, j])
                for k = i + 1:j - 1
                    Fij += (F[i, k]*T[k, j] - T[i, k]*F[k, j])/(T[i, i] - T[j, j])
                end
                F.data[i, j] = Fij
            end
        end
        F
    end
    matmap(f, T::UpperTriangular) = matmap!(UpperTriangular(zeros(eltype(T), size(T))), f, T)

    for (f, fmb, fm) in ((:exp, :expm!, :expm), (:log, :logm!, :logm), (:sqrt, :sqrtm!, :sqrtm), (:sin, :sinm!, :sinm), (:cos, :cosm!, :cosm))
        @eval begin
            function $(fmb)(A::StridedMatrix)
                if istriu(A)
                    return convert(Matrix, matmap($f, UpperTriangular(A)))
                end
                sf = schurfact!(A)
                F = matmap($f, sf.T)
                return (sf.Z*F)*sf.Z'
            end

            $(fm)(A::StridedMatrix) = $(fmb)(copy(A))
        end
    end
end