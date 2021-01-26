
using Distributions
using Random
using LinearAlgebra
using ProgressMeter
E = LinearAlgebra.I
eye(n) = Matrix(1.0E,n,n) #identity matrix

#
#import Fontconfig, Cairo
using Gadfly

@doc """
    rlct_R(M,N,H,r)

Calculate the real log canonical threshold of reduced rank regression
when the input dimmension is M, the output one is N,
the learner's rank is H, and the true rank is r.

The arguments must be positive integers and H>=r must hold.

# Example
```julia-repl
julia> rlct_R(4,3,2,1)
4.0
```
"""
function rlct_R(M,N,H,r)
    H_0 = r
    @assert (M>=1)&(N>=1)&(H>=H_0)&(H_0>=0)
    if (N+H_0<=M+H) & (M+H_0<=N+H) & (H+H_0<=M+N)
        if (M+N+H+H_0)%2 == 0
            return (2*(H+H_0)*(M+N)-(M-N)^2-(H+H_0)^2)/8
        else
            return (2*(H+H_0)*(M+N)-(M-N)^2-(H+H_0)^2 +1)/8
        end
    elseif M+H<N+H_0
        return (M*H+N*H_0-H*H_0)/2
    elseif N+H<M+H_0
        return (N*H+M*H_0-H*H_0)/2
    elseif M+N<H+H_0
        return M*N/2
    else #must not occur!
        return -1
    end
end

@doc """
    rlct_LDA(M,N,H,r)

Calculate the real log canonical threshold of latent Dirichlet allocation
when the input dimmension is M, the output one is N,
the learner's rank is H, and the rank of the transformed matrix U_0V_0 is r.
The rigorous definition of r is stated in the paper.

M>=2 and N>=2 and H>=r+1>=1 must hold. 

# Example
```julia-repl
julia> rlct_LDA(4,3,2,1)
3.5
```
"""
function rlct_LDA(M,N,H,r)
    @assert (M>=2)&(N>=2)&(H>=r+1)&(r+1>=1)
    @assert (rlct_R(M,N,H,r+1)-N/2)==(rlct_R(M-1,N-1,H-1,r)+M/2-1/2)
    return rlct_R(M,N,H,r+1)-N/2
end

@doc """
    calc_r(A0, B0)

Calculate an intrinsic value of the true parameter as (A0, B0).
Transform from A0 and B0 to U0 and V0 and compute the rank of U_0V_0.
"""
function calc_r(A0, B0)
    M, H0 = size(A0)
    H02, N = size(B0)
    @assert H0==H02
    tildeA = A0[1:(M-1), 1:(H0-1)]
    a = A0[1:(M-1), H0]
    tildea = reduce(hcat, [a for _ in 1:(H0-1)])
    tildeB = B0[1:(H0-1), 1:(N-1)]
    b = B0[1:(H0-1), N]
    tildeb = reduce(hcat, [b for _ in 1:(N-1)])
    U0 = tildeA - tildea
    V0 = tildeB - tildeb
    return rank(U0*V0)
end


# fundamaental setting
## calc. RLCT
SIM_ITERS = 100
## MCMC
MCMC_K = 2000
BURNIN = 10000
THIN = 20
MCMC_ITER = BURNIN + THIN*MCMC_K
## hyperparameter in prior
ALPHA = 1.0
BETA = 1.0
## true and model
M = 4
N = 4
H_0 = 2
Random.seed!(1)
A_0 = hcat([rand(Dirichlet(ALPHA*ones(M))) for k in 1:H_0]...)
B_0 = hcat([rand(Dirichlet(BETA*ones(H_0))) for j in 1:N]...)
#### control variable (number of learner's topics)
H = 4
r = calc_r(A_0, B_0)


## data
SEED = 2020 #don't tune
L = 500 #number of words in each document
SAMPLESIZE = N*L
TESTSIZE = 200*SAMPLESIZE
## save path
LOG_FILE_PATH = joinpath("log","YYYYMMDD-LDA-H=$H-n=$SAMPLESIZE-T=$TESTSIZE.csv")
println(A_0)
println(B_0)

rlct_LDA(M,N,H,H_0)

@doc """
    generate_data(A::Array{Float64,2}, B::Array{Float64,2}; n::Int=SAMPLESIZE, seed=1)
    
    Generate artificiall data from the given LDA model like the following.
    for l=1:n
        generate words from Cat(x|AB[:,j]) for j=1,...,N
        as two format: docs array (N, n//N, M) and words array (n, M)
    end
    
    # Argments

    - A::Array{Float64,2}: stochastic matrix A= (the probability that the word is i when the topic is k)
      whose shape is (M,H).
    - B::Array{Float64,2}: stochastic matrix B= (the probability that the topic is k when the doc is j)
      whose shape is (H,N).
    - n::Int=SAMPLESIZE: the number of all words in all document. sample size.
    - seed::Int=1: the random seed.
    
    # Return
    - return: generated words (onehot(M) vectors) as two format
    - rtype: tuple(Array; shape is (N, n//N, M), Array; shape is (n, M))
"""
function generate_data(A::Array{Float64,2}, B::Array{Float64,2}; n::Int=SAMPLESIZE, seed=1)
    Random.seed!(seed);
    C = A*B
    M = size(C)[1]
    N = size(C)[2]
    L = div(n, N)
    @assert n%N==0
    words_arr = zeros(Bool,M,n)
    for j in 1:N
        words = rand(Multinomial(1, C[:,j]), L)
        words_arr[:,(1+(j-1)*L):(j*L)] = words
    end
    docs_arr = reshape(words_arr, (M,L,N))
    return (docs_arr, words_arr)    
end

function make_onehot(hot_ind::Int, dim::Int; binary_type=Bool)
    @assert hot_ind <= dim
    @assert 1 <= hot_ind
    v = zeros(binary_type, dim);
    v[hot_ind] = 1;
    return v
end

function get_doc_onehot(docs_arr::Array{Bool,3}; n::Int=SAMPLESIZE)
    #===
    :return: Array of onehot vectors which mean each document id
    :rtype: Array{Float64,2}; shape is (N, n). each column means the document id corresponding word.
    ===#
    N = size(docs_arr)[3]
    L = div(n, N)
    doc_onehot = hcat([hcat([make_onehot(j, N) for l in 1:L]...) for j in 1:N]...)
    return doc_onehot
end

# each sampling step for Gibbs sampler
@doc """
    topic_step(docs_arr::Array{Bool,3}, A::Array{Float64,2}, B::Array{Float64,2}; n=SAMPLESIZE)

Generate a hidden variable (i.e. topic) in Gibbs sampler.

# Arguments
- docs_arr::Array{Bool,3}: data as documents form whose shape is (M, L==n//N, N).
- A::Array{Float64,2}: parameter matrix which represents word appearance probability for each topic.
- B::Array{Float64,2}: parameter matrix which represents topic propotion for each document.
- n::Int=SAMPLESIZE: sample size i.e. number of words.

# Return
- return: Bool topic matrix Y whose shape is (H, n).
"""
function topic_step(docs_arr::Array{Bool,3}, A::Array{Float64,2}, B::Array{Float64,2}; n=SAMPLESIZE)
    H = size(A)[2]
    N = size(B)[2]
    L = div(n, N)
    tmp = hcat([vcat(reshape(sum(docs_arr.*log.(A[:,k]),dims=1), (L,N)) .+ log.(B[k,:])'...) for k in 1:H]...)
    tmp = exp.(tmp)
    eta = tmp ./ sum(tmp, dims=2)
    Y = Array{Bool}(hcat([rand(Multinomial(1,eta[l,:])) for l in 1:n]...))
    return Y
end


@doc """
    A_step(words_arr::Array{Bool,2}, Y::Array{Bool,2}, alpha; n=SAMPLESIZE)

Generate a stochastic matrix in Gibbs sampler.

# Arguments
- words_arr::Array{Bool,2}: data as words form whose shape is (M, n).
- Y::Array{Bool,2}: topic matrix whose shape is (H, n).
- alpha: hyperparameter of Dirichlet prior for A.
- n::Int=SAMPLESIZE: sample size i.e. number of words.

# Return
- return: stochastic matrix which represents word appearance probability for each topic.
"""
function A_step(words_arr::Array{Bool,2}, Y::Array{Bool,2}, alpha; n=SAMPLESIZE)
    H = size(Y)[1]
    M = size(words_arr)[1]
    #use_words = reshape(words_arr, (n, M))
    hat_alpha = (Y*words_arr')' .+ alpha
    A = hcat([rand(Dirichlet(hat_alpha[:,k])) for k in 1:H]...)
    return A
end


@doc """
    B_step(doc_onehot::Array{Bool,2}, Y::Array{Bool,2}, beta; n=SAMPLESIZE)

Generate a stochastic matrix in Gibbs sampler.

# Arguments
- docs_arr::Array{Bool,3}: data as documents form whose shape is (M, L==n//N, N).
- Y::Array{Bool,2}: topic matrix whose shape is (H, n).
- beta: hyperparameter of Dirichlet prior for B.
- n::Int=SAMPLESIZE: sample size i.e. number of words.

# Return
- return: stochastic matrix which topic propotion for each document.
"""
function B_step(doc_onehot::Array{Bool,2}, Y::Array{Bool,2}, beta; n=SAMPLESIZE)
    #===
    (topic x num_of_docs) stochastic matrix sampling step.
    
    :param docs_arr: data as documents form; shape is (M, L=n//N, N)
    :param Y: topic matrix; shape is (H, n)
    :param beta: hyperparameter of Dirichlet prior for B
    :type beta: T(which means scalar) or Array{T,1}(which means 1-d vector); length is N. Where T is Int or Float.
    :return: param matrix B
    ===#
    H = size(Y)[1]
    N = size(doc_onehot)[1]
    #useZ = get_doc_onehot(docs_arr)'
    hat_beta = Y*(doc_onehot') .+ beta
    B = hcat([rand(Dirichlet(hat_beta[:,j])) for j in 1:N]...)
    return B
end

@doc """
    model_pmf(x::Array{Bool,1}, z::Array{Bool,1}, A::Array{Float64,2}, B::Array{Float64,2})

Calculate a probability mass for given data and parameter.

# Arguments
- x::Array{Bool,1}: M-dim onehot vector which means a word.
- z::Array{Bool,1}: N-dim onehot vector which means a document.
- A::Array{Float64,2}: (M,H) stochastic matrix
  whose (i,k) element means a probability that the word is i when the topic is k.
- B::Array{Float64,2}: (H,N) stochastic matrix
  whose (k,j) element means a probability that the topic is k when the document is j.
"""
function model_pmf(x::Array{Bool,1}, z::Array{Bool,1}, A::Array{Float64,2}, B::Array{Float64,2})
    #===
    probability mass function of model p(x|z,A,B).
    See also the above markdown cell.
    
    :param x: an M-dim onehot vector which means a word.
    :param x: an N-dim onehot vector which means a document.
    :param A: an (M,H) stochastic matrix whose (i,k) element means a probability that the word is i when the topic is k.
    :param B: an (H,N) stochastic matrix whose (k,j) element means a probability that the topic is k when the document is j.
    ===#
    @assert sum(z) == 1
    @assert sum(x) == 1
    @assert size(A)[2] == size(B)[1]
    j = argmax(z) #z_j=1, z_{j'}=0 (j' != j)
    i = argmax(x) #x_i=1, x_{i'}=0 (i' != i)
    AB = A*B
    return AB[i,j]
end

# Gibbs sampler
function run_all_sampling!(
        docs_arr::Array{Bool,3}, words_arr::Array{Bool,2}, doc_onehot::Array{Bool,2}, init_Y::Array{Bool,2},
        allYs::Array{Bool,3}, allAs::Array{Float64,3}, allBs::Array{Float64,3})
    iters = size(allYs)[3];
    alpha = ALPHA;
    beta = BETA;
    sampling_progress = Progress(iters);
    Y = init_Y;
    @time for k in 1:iters
        ## parameter matrix sampling and saving
        A = A_step(words_arr, Y, alpha);
        B = B_step(doc_onehot, Y, beta);
        allAs[:,:,k] = A;
        allBs[:,:,k] = B;
        ## hidden variable (topic indicator) sampling and saving
        Y = topic_step(docs_arr, A, B);
        allYs[:,:,k] = Y;
        ## Progress update
        next!(sampling_progress);
    end
    #return (allYs, allAs, allBs)
end

#===
function calc_likelihoodMat!(
        k::Int, words_arr::Array{Bool,2}, doc_onehot::Array{Bool,2}, A::Array{Float64,2}, B::Array{Float64,2},
        likelihoodMat::Array{Float64,2}; n::Int=SAMPLESIZE)
    #===
    ===#
    ## generated quantity: likelihood matrix
    for l in 1:n
        x = words_arr[:,l];
        z = doc_onehot[:,l];
        likelihoodMat[l,k] = model_pmf(x,z,A,B);
    end
    return likelihoodMat
end
===#

function run_thining!(
        words_arr::Array{Bool,2}, doc_onehot::Array{Bool,2}, 
        allYs::Array{Bool,3}, allAs::Array{Float64,3}, allBs::Array{Float64,3},
        gsYs::Array{Bool,3}, gsAs::Array{Float64,3}, gsBs::Array{Float64,3}, likelihoodMat::Array{Float64,2};
        K::Int=MCMC_K, burn::Int=BURNIN, th::Int=THIN, n::Int=SAMPLESIZE)
    ## burnin and thining
    K = size(gsYs)[3]
    thining_progress = Progress(K);
    @time for k in 1:K
        A = allAs[:,:,burn+k*th];
        gsAs[:,:,k] = A;
        B = allBs[:,:,burn+k*th];
        gsBs[:,:,k] = B;
        gsYs[:,:,k] = allYs[:,:,burn+k*th];
        ## generated quantity: likelihood matrix
        #likelihoodMat = calc_likelihoodMat!(k, words_arr, doc_onehot, A, B, likelihoodMat)
        for l in 1:n
            x = words_arr[:,l];
            z = doc_onehot[:,l];
            likelihoodMat[l,k] = model_pmf(x,z,A,B);
        end
        next!(thining_progress);
    end
    #return (gsYs, gsAs, gsBs, likelihoodMat)
end

function run_Gibbs_sampler_cored(
    docs_arr::Array{Bool,3}, words_arr::Array{Bool,2}; K::Int=MCMC_K, burn::Int=BURNIN, th::Int=THIN, n::Int=SAMPLESIZE)
    #===
    Version: separating core functions
    ===#
    ## initial value of A and B is sampled from the prior distributuion
    #### Assume that prior is SYMMETRIC Dirichlet distribution. Hyparam is scalar. 
    init_A = hcat([rand(Dirichlet(M, ALPHA)) for k in 1:H]...);
    init_B = hcat([rand(Dirichlet(H, BETA)) for j in 1:N]...);
    init_Y = topic_step(docs_arr, init_A, init_B);
    ## sampling iteration
    #### allocate tensors for MCMC sample
    iters = burn + K*th;
    allAs = zeros(Float64, M, H, iters);
    gsAs = zeros(Float64, M, H, K);
    allBs = zeros(Float64, H, N, iters);
    gsBs = zeros(Float64, H, N, K);
    allYs = zeros(Bool, H, n, iters);
    gsYs = zeros(Bool, H, n, K);
    #### all sampling
    doc_onehot = get_doc_onehot(docs_arr);
    println("Start $iters iteration for GS")
    run_all_sampling!(docs_arr, words_arr, doc_onehot, init_Y, allYs, allAs, allBs)
    ## burnin and thining
    likelihoodMat = zeros(Float64, n, K)
    println("Start burn-in and thining from $iters to $K")
    run_thining!(words_arr, doc_onehot, allYs, allAs, allBs, gsYs, gsAs, gsBs, likelihoodMat)
    return (gsYs, gsAs, gsBs, likelihoodMat)
end

@doc """
    run_Gibbs_sampler(docs_arr, words_arr, alpha, beta; K=MCMC_K, burn=BURNIN, th=THIN, n=SAMPLESIZE, seed_MCMC=2)

Run Gibbs sampler for LDA.
Global variable `M` , `H` , and `N` are used
as vocab. size, num. of topics, and num. of documents, respectively.
Note that rtype is tuple of tensor(Array{T,3}), not tuple of Array{Array{T,2}}.
"""
function run_Gibbs_sampler(docs_arr, words_arr, alpha, beta; K=MCMC_K, burn=BURNIN, th=THIN, n=SAMPLESIZE, seed_MCMC=2)
    #===
    Run Gibbs sampler for LDA.
    Global variable `M` , `H` , and `N` are used
    as vocab. size, num. of topics, and num. of documents, respectively.
    Note that rtype is tuple of tensor(Array{T,3}), not tuple of Array{Array{T,2}}.
    
    :param docs_arr: data as document format
    :param words_arr: data as word format
    :param alpha: hyperparameter of Dirichlet prior for A
    :param beta: hyperparameter of Dirichlet prior for B
    :return: sample of Y(topic indicator variable),A and B from GS and likelihood matrix
    :rtype: tuple(Array{Bool,3};shape(H,n,K), Array{Float64,3};shape(M,H,K), Array{Float64,3};shape(H,N,K), Array{Float64,2};shape(n,K))
    ===#
    Random.seed!(seed_MCMC);
    ## initial value of A and B is sampled from the prior distributuion
    #### Assume that prior is SYMMETRIC Dirichlet distribution. Hyparam is scalar. 
    init_A = hcat([rand(Dirichlet(M, alpha)) for k in 1:H]...);
    init_B = hcat([rand(Dirichlet(H, beta)) for j in 1:N]...);
    Y = topic_step(docs_arr, init_A, init_B);
    ## sampling iteration
    #### allocate tensors for MCMC sample
    iters = burn + K*th;
    allAs = zeros(Float64, M, H, iters);
    gsAs = zeros(Float64, M, H, K);
    allBs = zeros(Float64, H, N, iters);
    gsBs = zeros(Float64, H, N, K);
    allYs = zeros(Bool, H, n, iters);
    gsYs = zeros(Bool, H, n, K);
    #### all sampling
    doc_onehot = get_doc_onehot(docs_arr);
    println("Start $iters iteration for GS")
    sampling_progress = Progress(iters);
    @time for k in 1:iters
        ## parameter matrix sampling and saving
        A = A_step(words_arr, Y, alpha);
        B = B_step(doc_onehot, Y, beta);
        allAs[:,:,k] = A;
        allBs[:,:,k] = B;
        ## hidden variable (topic indicator) sampling and saving
        Y = topic_step(docs_arr, A, B);
        allYs[:,:,k] = Y;
        ## Progress update
        next!(sampling_progress);
    end
    ## burnin and thining
    likelihoodMat = zeros(Float64, n, K)
    println("Start burn-in and thining from $iters to $K")
    thining_progress = Progress(K);
    @time for k in 1:K
        A = allAs[:,:,burn+k*th];
        gsAs[:,:,k] = A;
        B = allBs[:,:,burn+k*th];
        gsBs[:,:,k] = B;
        gsYs[:,:,k] = allYs[:,:,burn+k*th];
        ## generated quantity: likelihood matrix
        for l in 1:n
            x = words_arr[:,l];
            z = doc_onehot[:,l];
            likelihoodMat[l,k] = model_pmf(x,z,A,B);
        end
        next!(thining_progress);
    end
    return (gsYs, gsAs, gsBs, likelihoodMat)
end

function run_prior_sampling(docs_arr, words_arr, alpha, beta; K=MCMC_K, burn=BURNIN, th=THIN, n=SAMPLESIZE, seed_MCMC=2)
    #===
    Run prior sampling K times.
    
    :return: sample of Y(topic indicator variable),A and B from GS and likelihood matrix
    :rtype: tuple(Array{Bool,3};shape(H,n,K), Array{Float64,3};shape(M,H,K), Array{Float64,3};shape(H,N,K), Array{Float64,2};shape(n,K))
    ===#
    Random.seed!(seed_MCMC);
    ## sampling iteration
    #### allocate tensors for MCMC sample
    gsAs = zeros(Float64, M, H, K);
    gsBs = zeros(Float64, H, N, K);
    gsYs = zeros(Bool, H, n, K);
    #### all sampling
    doc_onehot = get_doc_onehot(docs_arr);
    println("Start $K iteration for prior sampling (no thining)")
    sampling_progress = Progress(K);
    @time for k in 1:K
        ## parameter matrix sampling and saving
        A = hcat([rand(Dirichlet(M, alpha)) for k in 1:H]...);
        B = hcat([rand(Dirichlet(H, beta)) for j in 1:N]...);
        gsAs[:,:,k] = A;
        gsBs[:,:,k] = B;
        ## hidden variable (topic indicator) sampling and saving
        Y = topic_step(docs_arr, A, B);
        gsYs[:,:,k] = Y;
        ## Progress update
        next!(sampling_progress);
    end
    ## calc likelihood matrix
    likelihoodMat = zeros(Float64, n, K)
    println("Start calculating likelihood matrix")
    thining_progress = Progress(K);
    @time for k in 1:K
        A = gsAs[:,:,k];
        B = gsBs[:,:,k];
        ## generated quantity: likelihood matrix
        for l in 1:n
            x = words_arr[:,l];
            z = doc_onehot[:,l];
            likelihoodMat[l,k] = model_pmf(x,z,A,B);
        end
        next!(thining_progress);
    end
    return (gsYs, gsAs, gsBs, likelihoodMat)
end

# define functions for calculating RLCT
@doc """
    calc_functional_var(loglikeMat::Array{Float64,2})

Calculate a functional variance from given loglikelihood matrix.
"""
function calc_functional_var(loglikeMat::Array{Float64,2})
    #===
    Calculate functional variance from loglike matrix.
    
    :param loglikeMat: a matrix whose (l,k) element is log p(x_l|z_l,A_k,B_k)
    ===#
    n = size(loglikeMat)[1]
    K = size(loglikeMat)[2]
    first_term = reshape(mean(loglikeMat.^2, dims=2),n)
    second_term = reshape(mean(loglikeMat, dims=2).^2, n)
    func_var = sum(first_term - second_term)
    return func_var
end

@doc """
    calc_predict_dist(x, z, As, Bs)

Calculate predictive distribution from given MCMC sample.
"""
function calc_predict_dist(x, z, As, Bs)
    #===
    Calculate pmf of predictive distribution.
    
    :param x: an M-dim onehot vector which means a word.
    :param z: an N-dim onehot vector which means a document.
    :param As: an Array whose [:,:,k] element means an MCMC sample of A.
    :param Bs: an Array whose [:,:,k] element means an MCMC sample of B.
    ===#
    @assert size(As)[3]==size(Bs)[3]
    K = size(As)[3]
    mass = 0.0
    for k in 1:K
        mass += model_pmf(x, z, As[:,:,k], Bs[:,:,k])
    end
    return mass/K
end

@doc """
    calc_normalized_WAIC(words_arr, doc_onehots, true_A, true_B, likelihoodMat)

Calculate normalized WAIC, i.e. emperical loss - WAIC.
"""
function calc_normalized_WAIC(words_arr, doc_onehots, true_A, true_B, likelihoodMat)
    #===
    Calculating normalized WAIC.
    ===#
    n = size(words_arr)[2]
    emp_loss = -mean(log.(mean(likelihoodMat, dims=2)))
    emp_entropy = -mean([log(model_pmf(words_arr[:,l], doc_onehots[:,l], true_A, true_B)) for l in 1:n])
    func_var = calc_functional_var(log.(likelihoodMat))
    #println(typeof(emp_entropy))
    normalized_WAIC = emp_loss - emp_entropy + func_var/n
    return normalized_WAIC
end

@doc """
    calc_generalization_error(true_A, true_B, As, Bs; nT=TESTSIZE, seed_T=3)

Calculate generalization error, i.e. KL divergence from the true distribution to the predictive one
using many test data generated by the true distribution.
"""
function calc_generalization_error(true_A, true_B, As, Bs; nT=TESTSIZE, seed_T=3)
    #===
    Calculating generalization error using test data generated by true distribution.
    ===#
    test_doc, test_words = generate_data(true_A, true_B, n=nT, seed=seed_T)
    test_doc_onehot = get_doc_onehot(test_doc, n=nT)
    ge = 0.0
    calc_gerr_progress = Progress(nT)
    println(nT)
    @time for t in 1:nT
        q = model_pmf(test_words[:,t], test_doc_onehot[:,t], true_A, true_B)
        pred = calc_predict_dist(test_words[:,t], test_doc_onehot[:,t], As, Bs)
        ge += log(q) - log(pred) #log(q/pred), but q and pred is small number.
        next!(calc_gerr_progress)
    end
    ge /= nT
    return ge
end


@doc """
    run_single_inference(true_A, true_B, seed)

Carry out an inference.
- Generate sample.
- Run Gibbs sampler and make the posterior distribution.
- Calculate (generalization error, normalized WAIC) (this is the return).
"""
function run_single_inference(true_A, true_B, seed)
    #===
    Run single inference.
    
    Random seed of each module is determined by the following rule:
    generate train data: seed + 1
    run Gibbs sampler: seed + 2
    generate test data: seed + 3
    
    Macro seed (SEED) must be disjoint from 3 and larger than 10.
    ===#
    Random.seed!(seed);
    train_X_docs, train_X_words = generate_data(true_A, true_B, n=SAMPLESIZE, seed=seed+1)
    train_Z = get_doc_onehot(train_X_docs, n=SAMPLESIZE)
    println("Gibbs Sampling")
    topic_Ys, param_As, param_Bs, likelihoodMat = run_Gibbs_sampler(train_X_docs, train_X_words, ALPHA, BETA, seed_MCMC=seed+2)
    #topic_Ys, param_As, param_Bs, likelihoodMat = run_Gibbs_sampler_cored(train_X_docs, train_X_words)
    println("Calculation Normalized WAIC")
    normalized_WAIC = calc_normalized_WAIC(train_X_words, train_Z, true_A, true_B, likelihoodMat)
    println("Calculaton Generalization Error")
    ge = calc_generalization_error(true_A, true_B, param_As, param_Bs, seed_T=seed+3)
    return (ge, normalized_WAIC)
end

function run_single_inference_debug(true_A, true_B, seed)
    #===
    Run single inference for debug mode.
    ===#
    Random.seed!(seed);
    train_X_docs, train_X_words = generate_data(true_A, true_B, n=SAMPLESIZE, seed=seed+1)
    train_Z = get_doc_onehot(train_X_docs, n=SAMPLESIZE)
    println("Gibbs Sampling")
    topic_Ys, param_As, param_Bs, likelihoodMat = run_Gibbs_sampler(train_X_docs, train_X_words, ALPHA, BETA, seed_MCMC=seed+2)
    #topic_Ys, param_As, param_Bs, likelihoodMat = run_Gibbs_sampler_cored(train_X_docs, train_X_words)
    println("Calculation Normalized WAIC")
    normalized_WAIC = calc_normalized_WAIC(train_X_words, train_Z, true_A, true_B, likelihoodMat)
    println("Calculaton Generalization Error")
    ge = calc_generalization_error(true_A, true_B, param_As, param_Bs, seed_T=seed+3)
    return (topic_Ys, param_As, param_Bs, likelihoodMat, ge, normalized_WAIC)
end

function run_single_inference_PS(true_A, true_B, seed)
    #===
    Run single inference using prior sampling (worst benchmark).
    ===#
    Random.seed!(seed);
    train_X_docs, train_X_words = generate_data(true_A, true_B, n=SAMPLESIZE, seed=seed+1)
    train_Z = get_doc_onehot(train_X_docs, n=SAMPLESIZE)
    println("Prior Sampling")
    topic_Ys, param_As, param_Bs, likelihoodMat = run_prior_sampling(train_X_docs, train_X_words, ALPHA, BETA, seed_MCMC=seed+2)
    #topic_Ys, param_As, param_Bs, likelihoodMat = run_Gibbs_sampler_cored(train_X_docs, train_X_words)
    println("Calculation Normalized WAIC")
    normalized_WAIC = calc_normalized_WAIC(train_X_words, train_Z, true_A, true_B, likelihoodMat)
    println("Calculaton Generalization Error")
    ge = calc_generalization_error(true_A, true_B, param_As, param_Bs, seed_T=seed+3)
    return (topic_Ys, param_As, param_Bs, likelihoodMat, ge, normalized_WAIC)
end

@doc """
    run_multi_inference(true_A, true_B, sim_iters, log_file_path)

Independently repeat inferences.
"""
function run_multi_inference(true_A, true_B, sim_iters, log_file_path)
    #===
    for it in 1:sim_iters
        run_single_inference
    end
    and calculate RLCT from above $sim_iters simulation results.
    
    :param true_A: true parameter stochastic matrix A
    :param true_B: true parameter stochastic matrix B
    :param sim_iters: number of simulations
    :param log_file_path: path of the file to write the experimental log
    :return: generalization errors and normalized WAICs in the simulations
    :rtype: tuple(Array{Float64,1}, Array{Float64,1})
    ===#
    seeds = SEED .+ SEED * (1:sim_iters)
    gerrors = zeros(Float64, sim_iters)
    normalized_WAICs = zeros(Float64, sim_iters)
    open(log_file_path, "a") do fp
        println(fp, "## Simulation Setting")
        println(fp, "M,N,H,H_0,L,MCMC_K,BURNIN,THIN,TESTSIZE,SIM_ITERS,SEED")
        println(fp, "$M,$N,$H,$H_0,$L,$MCMC_K,$BURNIN,$THIN,$TESTSIZE,$sim_iters,$SEED")
        println(fp, "## Simulation Log")
        println(fp, "iter,gerror,normalized_WAIC,RLCT,RLCT_SEM")
    end
    simulation_progress = Progress(sim_iters)
    for it in 1:sim_iters
        println("# start $it th simulation")
        ge, norm_W = run_single_inference(true_A, true_B, seeds[it])
        gerrors[it] = ge
        normalized_WAICs[it] = norm_W
        now_ges = gerrors[1:it]
        now_nwaics = normalized_WAICs[1:it]
        rlct = (SAMPLESIZE/2)*mean(now_ges + now_nwaics)
        rlct_sem = (SAMPLESIZE/2)*std(now_ges + now_nwaics)/sqrt(it)
        open(log_file_path, "a") do fp
            println(fp,"$it,$ge,$norm_W,$rlct,$rlct_sem")
            println("$it,$ge,$norm_W,$rlct,$rlct_sem")
        end
        next!(simulation_progress)
        #@assert ge>=0
        #@assert norm_W>=0
    end
    return (gerrors, normalized_WAICs)
end

# main process
##===
gerrors, normalized_WAICs = @time run_multi_inference(A_0, B_0, SIM_ITERS, LOG_FILE_PATH)
ge_and_nW = gerrors + normalized_WAICs
each_lams = (SAMPLESIZE/2) .* ge_and_nW
lam = (SAMPLESIZE/2)*mean(ge_and_nW)
lam_sd = (SAMPLESIZE/2)*std(ge_and_nW)/sqrt(SIM_ITERS)
println("The numerical RLCT is $lam ± $lam_sd")
#===#

println("$M, $N, $H, $r")
λ = rlct_LDA(M,N,H,r)
println("The exact RLCT = $λ")

using DelimitedFiles
each_lams_savepath = split(LOG_FILE_PATH, ".")[1]*"_each_lams.csv"
writedlm(each_lams_savepath,  each_lams, ',')

each_lams_mean = mean(each_lams)
each_lams_std = std(each_lams)
upperline = each_lams_mean + each_lams_std
middleline = each_lams_mean
lowerline = each_lams_mean - each_lams_std
trueline = λ
plot(y=each_lams, yintercept = [upperline, middleline, lowerline, trueline],
    Geom.violin, Geom.hline(color=["orange","red","orange","blue"], style=[:dot,:dash,:dot,:solid]))
