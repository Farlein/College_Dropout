



```
Load packages
```

using Random, Distributions

using Plots, StatsPlots

using CSV, DataFrames

using FreqTables, StatsBase

using Flux

using ProgressMeter

using MultivariateStats

using MLUtils


```
Load data
```

raw = CSV.read("Ret_Surv_w_fold.csv", DataFrame)

names(raw)

n_model = 30

var_list = [:FT_Ind, :US_Ind, :D_Major_Ind, :cum_gpa, :DFW_Count_last, :Listen_Count_last, :Acad_stndng_Last, :Tot_Credit_Rate, :cur_regis_rate, :Tot_Credit_MinorTerm_Rate, :TOT_TRNSFR_rate, :TOT_TEST_CREDIT_rate, :hs_gpa
            , :fed_efc_rate, :total_grant_rate, :parent_loan_rate, :student_loan_rate, :work_study_rate, :Balance_Rate
            , :Pell_Ind, :Fst_Gen_Ind, :Gender_Ind, :Resid_Ind, :age, :Eth_ASIAN_Ind, :Eth_BLACK_Ind, :Eth_HISPA_Ind, :Eth_WHITE_Ind]
outcome_list = [:Outcome_23_Ind]

n_var = length(var_list)

```
Loss function
```

function surv_loss(result, label, w)

    loss_hat = (w .* label .* vec(log.(result))
                .- vec(result) .*1
                
    )

    -mean(loss_hat)            
end


```
Model
```

Join(combine, paths) = Parallel(combine, paths)
Join(combine, paths...) = Join(combine, paths)

Model_FNN(node_num_1) = Chain(
    Join(hcat,
          Chain(Dense(n_var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, exp)),
          Chain(Dense(n_var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, exp)),
          Chain(Dense(n_var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, exp)),
          Chain(Dense(n_var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, exp)),
          Chain(Dense(n_var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, exp)),
          Chain(Dense(n_var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, exp))
        )

)


```
CV Loop to find the best w and Num_Node_1
```

for test_term in [2223:-10:2183;]
    train_term = test_term - 5
    data_train = raw[ (raw.academic_term .<test_term) .& (raw.Term_Seq .<=6) .& (raw.fold .!= 10), :] 

    data_train.Period_length .= 1
    data_train.Outcome_23_Ind = data_train.Outcome_2_Ind .+ data_train.Outcome_3_Ind

    n_fit = nrow(data_train)

    ### Find the best w

    w_list = [1.1:0.1:1.9;]
    NN_list = []

    for fold in 1:9 # there is only 9 folds in the training dataset
        node_num_1 = 11
        w = w_list[fold]

        data_vldt = data_train[ data_train.fold .!= fold, :]

        x_surv = (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , var_list]))))
        )
        y_surv = (transpose(Array((data_vldt[ data_vldt.Term_Seq .== 1 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 2 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 3 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 4 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 5 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 6 , outcome_list])))
        )
        y_flat = vcat(y_surv[1][1,:], y_surv[2][1,:], y_surv[3][1,:], y_surv[4][1,:], y_surv[5][1,:], y_surv[6][1,:])
 
        model = Model_FNN(node_num_1)

        opt_state = Flux.setup(Flux.Adam(0.01), model)

        losses = Float32[]

        @showprogress for epoch in 1:10000

            input = x_surv
            label = y_flat
    
            val, grads = Flux.withgradient(model) do m
              result = m(input)
              surv_loss(result, label, w)
            end
    
            push!(losses, val)
    
            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
              @warn "loss is $val on $epoch epochs" epoch
              continue
            end
    
            Flux.update!(opt_state, model, grads[1])
    
          # Stop training when some criterion is reached
            if  (length(losses) > 2) && (abs(losses[length(losses)-1] - losses[length(losses)]) <1e-7)
              println("stopping after $epoch epochs")
              break
            end
        end

        append!(NN_list, model)

    end    

    # Validate the models    

    vldt_F_score = zeros(9)

    for fold in 1:9
        data_vldt = data_train[ data_train.fold .== fold, :]

        n_vldt = nrow(data_vldt)
        data_vldt.Outcome_23_Ind = data_vldt.Outcome_2_Ind .+ data_vldt.Outcome_3_Ind

        input = (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , var_list]))))
        )
        y_surv = (transpose(Array((data_vldt[ data_vldt.Term_Seq .== 1 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 2 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 3 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 4 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 5 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 6 , outcome_list])))
        )
        label = vec(transpose(hcat(y_surv[1],y_surv[2],y_surv[3],y_surv[4],y_surv[5],y_surv[6])))

        model= NN_list[fold]

        h_vldt_fit= model(input)
        P_vldt_fit = 1 .- exp.(-h_vldt_fit .*1)
        θ_vldt_fit = P_vldt_fit[1,:]

        y_vldt_pred = zeros(n_vldt) 
        F_score = []

        for thres in [0.11:0.01:0.20;]
            for i in 1:n_vldt
                if θ_vldt_fit[i] >= thres
                    y_vldt_pred[i] =1
                else y_vldt_pred[i] = 0
                end
            end
        
            if maximum(y_vldt_pred) == 0
                append!(F_score, 0)
                println(thres, " ", 0)
                continue
            end

            vldt_conf_mtx = freqtable(label, y_vldt_pred)
        
            precise = vldt_conf_mtx[4]/(vldt_conf_mtx[4]+vldt_conf_mtx[3])
            recall = vldt_conf_mtx[4]/(vldt_conf_mtx[4]+vldt_conf_mtx[2])
            append!(F_score, 2*precise*recall/(precise+recall))
        
            display(vldt_conf_mtx)
            println(thres, " ", precise, " ", recall, " ", F_score)    

        end

        vldt_F_score[fold] = maximum(F_score)
    end

    w_results = DataFrame(w=w_list, F_score=vldt_F_score)
    w_select = w_results[w_results.F_score .== maximum(w_results.F_score), :w]

    ### Find the best node_num_1

    node_list = [9:17;]
    NN_list = []

    for fold in 1:9
        w = w_select
        node_num_1 = node_list[fold]

        data_vldt = data_train[ data_train.fold .!= fold, :]
        x_surv = (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , var_list]))))
        )
        y_surv = (transpose(Array((data_vldt[ data_vldt.Term_Seq .== 1 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 2 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 3 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 4 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 5 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 6 , outcome_list])))
        )
        y_flat = vec(transpose(hcat(y_surv[1],y_surv[2],y_surv[3],y_surv[4],y_surv[5],y_surv[6])))

        model = Model_FNN(node_num_1)

        opt_state = Flux.setup(Flux.Adam(0.01), model)

        losses = Float32[]

        @showprogress for epoch in 1:10000

            input = x_surv
            label = y_flat
    
            val, grads = Flux.withgradient(model) do m
              result = m(input)
              surv_loss(result, label, w)
            end
    
            push!(losses, val)
    
            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
              @warn "loss is $val on $epoch epochs" epoch
              continue
            end
    
            Flux.update!(opt_state, model, grads[1])
    
          # Stop training when some criterion is reached
            if  (length(losses) > 2) && (abs(losses[length(losses)-1] - losses[length(losses)]) <1e-7)
              println("stopping after $epoch epochs")
              break
            end
        end

        append!(NN_list, model)
    end

    # Validate the models

    vldt_F_score = zeros(9)

    for fold in 1:9
        data_vldt = data_train[ data_train.fold .== fold, :]

        n_vldt = nrow(data_vldt)
        data_vldt.Outcome_23_Ind = data_vldt.Outcome_2_Ind .+ data_vldt.Outcome_3_Ind

        input = (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , var_list]))))
        , transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , var_list]))))
        )
        y_surv = (transpose(Array((data_vldt[ data_vldt.Term_Seq .== 1 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 2 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 3 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 4 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 5 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 6 , outcome_list])))
        )
        label = vec(transpose(hcat(y_surv[1],y_surv[2],y_surv[3],y_surv[4],y_surv[5],y_surv[6])))

        model= NN_list[fold]

        h_vldt_fit= model(input)
        P_vldt_fit = 1 .- exp.(-h_vldt_fit .*1)
        θ_vldt_fit = P_vldt_fit[1,:]

        y_vldt_pred = zeros(n_vldt) 
        F_score = []

        for thres in [0.11:0.01:0.20;]
            for i in 1:n_vldt
                if θ_vldt_fit[i] >= thres
                    y_vldt_pred[i] =1
                else y_vldt_pred[i] = 0
                end
            end

            if maximum(y_vldt_pred) == 0
                append!(F_score, 0)
                println(thres, " ", 0)
                continue
            end

            vldt_conf_mtx = freqtable(label, y_vldt_pred)
        
            precise = vldt_conf_mtx[4]/(vldt_conf_mtx[4]+vldt_conf_mtx[3])
            recall = vldt_conf_mtx[4]/(vldt_conf_mtx[4]+vldt_conf_mtx[2])
            append!(F_score, 2*precise*recall/(precise+recall))
        
            display(vldt_conf_mtx)
            println(thres, " ", precise, " ", recall, " ", F_score)    

        end

        vldt_F_score[fold] = maximum(F_score)
    end

    Node_results = DataFrame(Num_Node_1=node_list, F_score=vldt_F_score)
    node_select = Node_results[Node_results.F_score .== maximum(Node_results.F_score), :Num_Node_1]    

    ### Summarize CV results
    CV_results = DataFrame(Test_Term = test_term, w = w_select, Num_Node_1=node_select)
    path = "G:\\My Drive\\FSAN\\9_Ret and Grad\\Temp results"
    file = "NN_CV_" * string(test_term) * ".csv"
    CSV.write(joinpath(path,file), CV_results)
    
end

```
Model
Neural Xβ
Fully connected
```

function my_loss(result, label)

    w_1 = 1.4

    loss_hat = (w_1 .* label .* vec(log.(result))
                .- vec(result) .*1
                
    )

    -mean(loss_hat)            
end

for test_term_2 in [2223:-10:2203;] 

    fit_term = test_term_2 - 5
    data_fit = raw[ (raw.academic_term .<=fit_term) .& (raw.Term_Seq .<=6) .& (raw.fold .!= 10),:]
    
    data_fit.Period_length .= 1
    data_fit.Outcome_23_Ind = data_fit.Outcome_2_Ind .+ data_fit.Outcome_3_Ind
 
    x_surv = (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 1 , var_list]))))
    , transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 2 , var_list]))))
    , transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 3 , var_list]))))
    , transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 4 , var_list]))))
    , transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 5 , var_list]))))
    , transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 6 , var_list]))))
    )

    y_surv = (transpose(Array((data_fit[ data_fit.Term_Seq .== 1 , outcome_list])))
            , transpose(Array((data_fit[ data_fit.Term_Seq .== 2 , outcome_list])))
            , transpose(Array((data_fit[ data_fit.Term_Seq .== 3 , outcome_list])))
            , transpose(Array((data_fit[ data_fit.Term_Seq .== 4 , outcome_list])))
            , transpose(Array((data_fit[ data_fit.Term_Seq .== 5 , outcome_list])))
            , transpose(Array((data_fit[ data_fit.Term_Seq .== 6 , outcome_list])))
            )
    y_flat = vcat(y_surv[1][1,:], y_surv[2][1,:], y_surv[3][1,:], y_surv[4][1,:], y_surv[5][1,:], y_surv[6][1,:])

    NN_list = []
    
    for model_num in 1:n_model

        node_num_1 = 11
        model = Model_FNN(node_num_1)

        opt_state = Flux.setup(Flux.Adam(0.01), model)

        losses = Float32[]

        @showprogress for epoch in 1:10000
            input = x_surv
            label = y_flat
        
            val, grads = Flux.withgradient(model) do m
              result = m(input)
              my_loss(result, label)
            end

            push!(losses, val)
        
            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
              @warn "loss is $val on $epoch epochs" epoch
              continue
            end
        
            Flux.update!(opt_state, model, grads[1])
        
          # Stop training when some criterion is reached
          if  (length(losses) > 2) && (abs(losses[length(losses)-1] - losses[length(losses)]) <1e-7)
            println("stopping after $epoch epochs")
            break
          end
        end

        append!(NN_list, model)

    end

    using BSON: @save
    for i in 1:n_model
        model = NN_list[i]
        path = "NN_" * string(test_term_2) * "_" * string(i) * ".bson"
        @save path model
    end    

end

```
Find threshold probabilities for a specific test semester
```

### split into training, hold-out and test datasets
fit_term = 2218
data_fit = raw[ (raw.academic_term .<=fit_term) .& (raw.Term_Seq .<=6) .& (raw.fold .!= 10),:]
data_flow = raw[ (raw.academic_term .<=fit_term) .& (raw.Term_Seq .<=6) .& (raw.fold .== 10),:]

test_term_2 = 2223
data_test = raw[ (raw.academic_term .==test_term_2) .& (raw.Term_Seq .<=6), :]

#####
data_fit.Outcome_23_Ind = data_fit.Outcome_2_Ind .+ data_fit.Outcome_3_Ind

x_surv = (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 1 , var_list]))))
, transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 2 , var_list]))))
, transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 3 , var_list]))))
, transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 4 , var_list]))))
, transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 5 , var_list]))))
, transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 6 , var_list]))))
)

y_acutal = (transpose(Array((data_fit[ data_fit.Term_Seq .== 1 , :Outcome_23_Ind])))
        , transpose(Array((data_fit[ data_fit.Term_Seq .== 2 , :Outcome_23_Ind])))
        , transpose(Array((data_fit[ data_fit.Term_Seq .== 3 , :Outcome_23_Ind])))
        , transpose(Array((data_fit[ data_fit.Term_Seq .== 4 , :Outcome_23_Ind])))
        , transpose(Array((data_fit[ data_fit.Term_Seq .== 5 , :Outcome_23_Ind])))
        , transpose(Array((data_fit[ data_fit.Term_Seq .== 6 , :Outcome_23_Ind])))
        )
y_acutal_flat = transpose(hcat(y_acutal[1],y_acutal[2],y_acutal[3],y_acutal[4],y_acutal[5],y_acutal[6]))

#####
n_test = nrow(data_test)
data_test.Outcome_23_Ind = data_test.Outcome_2_Ind .+ data_test.Outcome_3_Ind

x_test_surv = (transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 1 , var_list]))))
, transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 2 , var_list]))))
, transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 3 , var_list]))))
, transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 4 , var_list]))))
, transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 5 , var_list]))))
, transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 6 , var_list]))))
)

y_test_acutal = (transpose(Array((data_test[ data_test.Term_Seq .== 1 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 2 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 3 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 4 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 5 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 6 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 7 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 8 , :Outcome_23_Ind]))))

y_test_acutal_flat = transpose(hcat(y_test_acutal[1],y_test_acutal[2],y_test_acutal[3],y_test_acutal[4],y_test_acutal[5],y_test_acutal[6]))

#####
seq_list = []
for i in 1:6
    append!(seq_list, i*ones(sum(data_fit.Term_Seq .== i)))
end

lens = zeros(3)
for i in 1:3
    lens[i] = size(y_test_acutal[i*2])[2]
end
lens = Int.(lens)

### Load trained models
NN_list = []
using BSON: @load
for i in 1:n_model
    path = "NN_" * string(test_term_2) * "_" * string(i) * ".bson" 
    @load path model
    push!(NN_list, model)
end

### find threshold probabilities

thres_slcts = allowmissing([])
test_F_scores = allowmissing([])
y_test_list = allowmissing([])

for k in 1:n_model
    println("k=: ", k)
    model= NN_list[k]

    out2 = model(x_surv)
    P_out = 1 .- exp.(-out2 .*1)
    θ_1_raw = P_out[1,:]

    thres_list = [0.15:0.01:0.50;]
    thres_slct = []

    for j in 1:6
        θ_pred = θ_1_raw[seq_list .== convert(AbstractFloat, j)]

        F_scores = []
        for i in 1:length(thres_list)
            y_pred = θ_pred .>= thres_list[i]

            if maximum(y_pred) == 0 
                push!(F_scores, missing)
                continue
            end

            if minimum(y_pred) == 1
                push!(F_scores, missing)
                continue
            end

            fit_conf_mtx = freqtable(vec(y_acutal[j]), y_pred)
            fit_precise = fit_conf_mtx[4]/(fit_conf_mtx[4]+fit_conf_mtx[3])
            fit_recall = fit_conf_mtx[4]/(fit_conf_mtx[4]+fit_conf_mtx[2])
            fit_F_score = 2*fit_precise*fit_recall/(fit_precise+fit_recall)
            push!(F_scores, fit_F_score)

        end

        push!(thres_slct,thres_list[findmax(F_scores)[2]])
    end

    push!(thres_slcts , thres_slct)

    h_test_fit = model(x_test_surv)
    P_test_fit = 1 .- exp.(-h_test_fit .*1)    
    θ_test_fit = P_test_fit[1,:]

    thres_slct_list = vcat(ones(lens[1])*thres_slct[2], ones(lens[2])*thres_slct[4], ones(lens[3])*thres_slct[6])
    y_test_pred = θ_test_fit .>= thres_slct_list
    push!(y_test_list, y_test_pred)
    
    if maximum(y_test_pred) == 0
        push!(test_F_scores, 0)
        continue
    end    

    test_conf_mtx = freqtable(y_test_acutal_flat, y_test_pred)
    test_precise = test_conf_mtx[4]/(test_conf_mtx[4]+test_conf_mtx[3])
    test_recall = test_conf_mtx[4]/(test_conf_mtx[4]+test_conf_mtx[2])
    test_F_score = 2*test_precise*test_recall/(test_precise+test_recall)
    display(test_conf_mtx)
    println( " ",test_precise, " ", test_recall, " ", test_F_score)
    push!(test_F_scores, test_F_score)

end

thres_matrix = transpose(hcat(thres_slcts...))
thres_results = DataFrame(thres_1 = thres_matrix[:,1], thres_2 = thres_matrix[:,2], thres_3 = thres_matrix[:,3], thres_4 = thres_matrix[:,4], thres_5 = thres_matrix[:,5], thres_6 = thres_matrix[:,6], test_F_scores = test_F_scores)

path = "G:/My Drive/FSAN/9_Ret and Grad/Results/NN_Threshold_" * string(test_term_2) * ".csv"
CSV.write(path, thres_results)

```
Caluclate F1-score for the test dataset
```

y_test_matrix = hcat(y_test_list...)
y_test_df = DataFrame(y_test_matrix, :auto)
y_test_df.y_actual = y_test_acutal_flat

y_hat = vec(mean(y_test_matrix, dims = 2) .> 0.5)
conf_mtx = freqtable(y_test_acutal_flat, y_hat)
precise = conf_mtx[4]/(conf_mtx[4]+conf_mtx[3])
recall = conf_mtx[4]/(conf_mtx[4]+conf_mtx[2])
F_score = 2*precise*recall/(precise+recall)
