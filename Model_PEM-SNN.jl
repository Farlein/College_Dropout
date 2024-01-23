


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


Acad_Var_List = [:FT_Ind, :US_Ind, :D_Major_Ind, :cum_gpa, :DFW_Count_last, :Listen_Count_last, :Acad_stndng_Last, :Tot_Credit_Rate, :cur_regis_rate, :Tot_Credit_MinorTerm_Rate, :TOT_TRNSFR_rate, :TOT_TEST_CREDIT_rate, :hs_gpa]
Fin_Var_List = [:fed_efc_rate, :total_grant_rate, :parent_loan_rate, :student_loan_rate, :work_study_rate, :Balance_Rate]
Socio_Var_List = [:Pell_Ind, :Fst_Gen_Ind, :Gender_Ind, :Resid_Ind, :age, :Eth_ASIAN_Ind, :Eth_BLACK_Ind, :Eth_HISPA_Ind, :Eth_WHITE_Ind]

n_Acad_Var = length(Acad_Var_List)
n_Fin_Var = length(Fin_Var_List)
n_Socio_Var = length(Socio_Var_List)

n_var = n_Acad_Var+n_Fin_Var+n_Socio_Var

outcome_list = [:Outcome_23_Ind]

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

Model_SNN(node_num_1) = Chain(
                Join(hcat,
                Chain(
                    Join(vcat,
                        Chain(Dense(n_Acad_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Fin_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Socio_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast))
                    ),
                    Dense(3 => 1, exp)
                    ),
                Chain(
                    Join(vcat,
                        Chain(Dense(n_Acad_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Fin_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Socio_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast))
                    ),
                    Dense(3 => 1, exp)
                    ),
                Chain(
                    Join(vcat,
                        Chain(Dense(n_Acad_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Fin_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Socio_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast))
                    ),
                    Dense(3 => 1, exp)
                    ),
                Chain(
                    Join(vcat,
                        Chain(Dense(n_Acad_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Fin_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Socio_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast))
                    ),
                    Dense(3 => 1, exp)
                    ),
                Chain(
                    Join(vcat,
                        Chain(Dense(n_Acad_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Fin_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Socio_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast))
                    ),
                    Dense(3 => 1, exp)
                    ),
                Chain(
                    Join(vcat,
                        Chain(Dense(n_Acad_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Fin_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast)),
                        Chain(Dense(n_Socio_Var => node_num_1, sigmoid_fast; init=Flux.zeros32),Dense(node_num_1 => 1, sigmoid_fast))
                    ),
                    Dense(3 => 1, exp)
                    )
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

        x_surv = (
            (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Socio_Var_List])))))
        )
        y_surv = (transpose(Array((data_vldt[ data_vldt.Term_Seq .== 1 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 2 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 3 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 4 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 5 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 6 , outcome_list])))
        )
        y_flat = vcat(y_surv[1][1,:], y_surv[2][1,:], y_surv[3][1,:], y_surv[4][1,:], y_surv[5][1,:], y_surv[6][1,:])
 
        model = Model_SNN(node_num_1)

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

        input = (
            (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Socio_Var_List])))))
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
        x_surv = (
            (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Socio_Var_List])))))
        )
        y_surv = (transpose(Array((data_vldt[ data_vldt.Term_Seq .== 1 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 2 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 3 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 4 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 5 , outcome_list])))
        , transpose(Array((data_vldt[ data_vldt.Term_Seq .== 6 , outcome_list])))
        )
        y_flat = vec(transpose(hcat(y_surv[1],y_surv[2],y_surv[3],y_surv[4],y_surv[5],y_surv[6])))

        model = Model_SNN(node_num_1)

        opt_state = Flux.setup(Flux.Adam(0.01), model)

        losses = Float32[]

        @showprogress for epoch in 1:10000
            #val = 0

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
    
          # Compute some accuracy, and save details as a NamedTuple
          #acc = my_accuracy(model, train_set)
          #push!(my_log, (; acc, losses))
    
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

        input = (
            (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 1 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 2 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 3 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 4 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 5 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Acad_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Fin_Var_List])))), transpose(Float32.(Array((data_vldt[ data_vldt.Term_Seq .== 6 , Socio_Var_List])))))
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
    file = "SNN_CV_" * string(test_term) * ".csv"
    CSV.write(joinpath(path,file), CV_results)

end

```
Model
Neural Xβ
Structural
```

function my_loss(result, label)

    w_1 = 1.3

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
 
    x_surv = (
            (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 1 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 1 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 1 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 2 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 2 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 2 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 3 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 3 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 3 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 4 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 4 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 4 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 5 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 5 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 5 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 6 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 6 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 6 , Socio_Var_List])))))
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
        model = Model_SNN(node_num_1)

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
        path = "SNN_" * string(test_term_2) * "_" * string(i) * ".bson"
        @save path model
    end

end

```
Find threshold probabilities for a specific test semester
```

### split into training, hold-out and test datasets
fit_term = 2218
data_fit = raw[ (raw.academic_term .<=fit_term) .& (raw.Term_Seq .<=6) .& (raw.fold .!= 10),:]
data_hold = raw[ (raw.academic_term .<=fit_term) .& (raw.Term_Seq .<=6) .& (raw.fold .== 10),:]

test_term_2 = 2223
data_test = raw[ (raw.academic_term .==test_term_2) .& (raw.Term_Seq .<=6), :]

#####
data_fit.Outcome_23_Ind = data_fit.Outcome_2_Ind .+ data_fit.Outcome_3_Ind

x_surv = (
            (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 1 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 1 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 1 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 2 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 2 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 2 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 3 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 3 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 3 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 4 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 4 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 4 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 5 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 5 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 5 , Socio_Var_List])))))
            , (transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 6 , Acad_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 6 , Fin_Var_List])))), transpose(Float32.(Array((data_fit[ data_fit.Term_Seq .== 6 , Socio_Var_List])))))
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

x_test_surv = (
        (transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 1 , Acad_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 1 , Fin_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 1 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 2 , Acad_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 2 , Fin_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 2 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 3 , Acad_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 3 , Fin_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 3 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 4 , Acad_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 4 , Fin_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 4 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 5 , Acad_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 5 , Fin_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 5 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 6 , Acad_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 6 , Fin_Var_List])))), transpose(Float32.(Array((data_test[ data_test.Term_Seq .== 6 , Socio_Var_List])))))
)

y_test_acutal = (transpose(Array((data_test[ data_test.Term_Seq .== 1 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 2 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 3 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 4 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 5 , :Outcome_23_Ind])))
        , transpose(Array((data_test[ data_test.Term_Seq .== 6 , :Outcome_23_Ind])))
        )

y_test_acutal_flat = transpose(hcat(y_test_acutal[1],y_test_acutal[2],y_test_acutal[3],y_test_acutal[4],y_test_acutal[5],y_test_acutal[6]))

#####
lens = zeros(3)
for i in 1:3
    lens[i] = size(y_test_acutal[i*2])[2]
end
lens = Int.(lens)

seq_list = []
for i in 1:6
    append!(seq_list, i*ones(sum(data_fit.Term_Seq .== i)))
end

### Load trained models
NN_list = []
using BSON: @load
for i in 1:n_model
    path = "SNN_" * string(test_term_2) * "_" * string(i) * ".bson" 
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
                #push!(thres_slct, missing)
                continue
            end

            if minimum(y_pred) == 1
                push!(F_scores, missing)
                #push!(thres_slct, missing)
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

path = "G:/My Drive/FSAN/9_Ret and Grad/Results/SNN_Threshold_" * string(test_term_2) * ".csv"
CSV.write(path, thres_results)

```
Caluclate F1-score for the test dataset
```

y_test_matrix = hcat(y_test_list...)
y_test_df = DataFrame(y_test_matrix, :auto)
y_test_df.y_actual = y_test_acutal_flat
mean(y_test_matrix, dims = 2)
sum(mean(y_test_matrix, dims = 2))
sum(mean(y_test_matrix, dims = 2) .> 0.5)

y_hat = vec(mean(y_test_matrix, dims = 2) .> 0.5)
conf_mtx = freqtable(y_test_acutal_flat, y_hat)
precise = conf_mtx[4]/(conf_mtx[4]+conf_mtx[3])
recall = conf_mtx[4]/(conf_mtx[4]+conf_mtx[2])
F_score = 2*precise*recall/(precise+recall)

```
Predictive Performance of Various Student Groups in the Test Dataset
```

id_test = (transpose(Array((data_test[ data_test.Term_Seq .== 1 , :student_id])))
, transpose(Array((data_test[ data_test.Term_Seq .== 2 , :student_id])))
, transpose(Array((data_test[ data_test.Term_Seq .== 3 , :student_id])))
, transpose(Array((data_test[ data_test.Term_Seq .== 4 , :student_id])))
, transpose(Array((data_test[ data_test.Term_Seq .== 5 , :student_id])))
, transpose(Array((data_test[ data_test.Term_Seq .== 6 , :student_id])))
)

id_test_flat = transpose(hcat(id_test[1],id_test[2],id_test[3],id_test[4],id_test[5],id_test[6]))
id_test_df = DataFrame(student_id = id_test_flat)

output_test_df = DataFrame(student_id = id_test_flat, y_hat = y_hat)

data_test_w_pred = innerjoin(data_test, output_test_df, on = :student_id)

### Term_Seq
for i in 1:3
    conf_mtx = freqtable(data_test_w_pred[data_test_w_pred.Term_Seq .== (2*i), :].Outcome_23_Ind, data_test_w_pred[data_test_w_pred.Term_Seq .== (2*i), :].y_hat)
    precise = conf_mtx[4]/(conf_mtx[4]+conf_mtx[3])
    recall = conf_mtx[4]/(conf_mtx[4]+conf_mtx[2])
    println("Term Seq = ", 2*i, ": F_score = ", 2*precise*recall/(precise+recall) )
    println(mean(data_test_w_pred[data_test_w_pred.Term_Seq .== (2*i), :].Outcome_23_Ind))
end

### Gender
conf_mtx = freqtable(data_test_w_pred[data_test_w_pred.Gender_Ind .== 1, :].Outcome_23_Ind, data_test_w_pred[data_test_w_pred.Gender_Ind .== 1, :].y_hat)
precise = conf_mtx[4]/(conf_mtx[4]+conf_mtx[3])
recall = conf_mtx[4]/(conf_mtx[4]+conf_mtx[2])
F_score = 2*precise*recall/(precise+recall)
println("Gender = Male", ": F_score = ", F_score )
println(mean(data_test_w_pred[data_test_w_pred.Gender_Ind .== 1, :].Outcome_23_Ind))

conf_mtx = freqtable(data_test_w_pred[data_test_w_pred.Gender_Ind .== 0, :].Outcome_23_Ind, data_test_w_pred[data_test_w_pred.Gender_Ind .== 0, :].y_hat)
precise = conf_mtx[4]/(conf_mtx[4]+conf_mtx[3])
recall = conf_mtx[4]/(conf_mtx[4]+conf_mtx[2])
F_score = 2*precise*recall/(precise+recall)
println("Gender = Female", ": F_score = ", F_score )
println(mean(data_test_w_pred[data_test_w_pred.Gender_Ind .== 0, :].Outcome_23_Ind))

### Pell
conf_mtx = freqtable(data_test_w_pred[data_test_w_pred.Pell_Ind .== 1, :].Outcome_23_Ind, data_test_w_pred[data_test_w_pred.Pell_Ind .== 1, :].y_hat)
precise = conf_mtx[4]/(conf_mtx[4]+conf_mtx[3])
recall = conf_mtx[4]/(conf_mtx[4]+conf_mtx[2])
F_score = 2*precise*recall/(precise+recall)
println("Pell = Yes", ": F_score = ", F_score )
println(mean(data_test_w_pred[data_test_w_pred.Pell_Ind .== 1, :].Outcome_23_Ind))

conf_mtx = freqtable(data_test_w_pred[data_test_w_pred.Pell_Ind .== 0, :].Outcome_23_Ind, data_test_w_pred[data_test_w_pred.Pell_Ind .== 0, :].y_hat)
precise = conf_mtx[4]/(conf_mtx[4]+conf_mtx[3])
recall = conf_mtx[4]/(conf_mtx[4]+conf_mtx[2])
F_score = 2*precise*recall/(precise+recall)
println("Pell = No", ": F_score = ", F_score )
println(mean(data_test_w_pred[data_test_w_pred.Pell_Ind .== 0, :].Outcome_23_Ind))

```
Flow of Dropout Risk in the Hold-Out Dataset
```

data_hold.Outcome_23_Ind = data_hold.Outcome_2_Ind .+ data_hold.Outcome_3_Ind

x_flow_surv = (
        (transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 1 , Acad_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 1 , Fin_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 1 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 2 , Acad_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 2 , Fin_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 2 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 3 , Acad_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 3 , Fin_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 3 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 4 , Acad_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 4 , Fin_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 4 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 5 , Acad_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 5 , Fin_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 5 , Socio_Var_List])))))
        , (transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 6 , Acad_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 6 , Fin_Var_List])))), transpose(Float32.(Array((data_hold[ data_hold.Term_Seq .== 6 , Socio_Var_List])))))
)

y_flow_acutal = (transpose(Array((data_hold[ data_hold.Term_Seq .== 1 , :Outcome_23_Ind])))
        , transpose(Array((data_hold[ data_hold.Term_Seq .== 2 , :Outcome_23_Ind])))
        , transpose(Array((data_hold[ data_hold.Term_Seq .== 3 , :Outcome_23_Ind])))
        , transpose(Array((data_hold[ data_hold.Term_Seq .== 4 , :Outcome_23_Ind])))
        , transpose(Array((data_hold[ data_hold.Term_Seq .== 5 , :Outcome_23_Ind])))
        , transpose(Array((data_hold[ data_hold.Term_Seq .== 6 , :Outcome_23_Ind])))
        )
y_flow_acutal_flat = transpose(hcat(y_flow_acutal[1],y_flow_acutal[2],y_flow_acutal[3],y_flow_acutal[4],y_flow_acutal[5],y_flow_acutal[6]))


lens = zeros(6)
for i in 1:6
    lens[i] = size(y_flow_acutal[i])[2]
end
lens = Int.(lens)

### Derive θ_hat and y_hat

thres_slcts = allowmissing([])
flow_F_scores = allowmissing([])
y_flow_list = allowmissing([])
θ_flow_list = allowmissing([])

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

    h_flow_fit = model(x_flow_surv)
    P_flow_fit = 1 .- exp.(-h_flow_fit .*1)    
    θ_flow_fit = P_flow_fit[1,:]

    push!(θ_flow_list, θ_flow_fit)

    thres_slct_list = vcat(ones(lens[1])*thres_slct[2], ones(lens[2])*thres_slct[2], ones(lens[3])*thres_slct[3], ones(lens[4])*thres_slct[4], ones(lens[5])*thres_slct[5], ones(lens[6])*thres_slct[6])
    y_flow_pred = θ_flow_fit .>= thres_slct_list
    push!(y_flow_list, y_flow_pred)
    
    if maximum(y_flow_pred) == 0
        push!(flow_F_scores, 0)
        continue
    end    

    flow_conf_mtx = freqtable(y_flow_acutal_flat, y_flow_pred)
    flow_precise = flow_conf_mtx[4]/(flow_conf_mtx[4]+flow_conf_mtx[3])
    flow_recall = flow_conf_mtx[4]/(flow_conf_mtx[4]+flow_conf_mtx[2])
    flow_F_score = 2*flow_precise*flow_recall/(flow_precise+flow_recall)
    display(flow_conf_mtx)
    println( " ",flow_precise, " ", flow_recall, " ", flow_F_score)
    push!(flow_F_scores, flow_F_score)

end

θ_flow_matrix = hcat(θ_flow_list...)
θ_hat = vec(mean(y_flow_matrix, dims = 2))

y_flow_matrix = hcat(y_flow_list...)
y_hat = vec(mean(y_flow_matrix, dims = 2) .> 0.5)

### Observe the predicted dropout probabilities from dropout students in the hold-out dataset

id_flow = (transpose(Array((data_hold[ data_hold.Term_Seq .== 1 , [:student_id, :Term_Seq]])))
, transpose(Array((data_hold[ data_hold.Term_Seq .== 2 , [:student_id, :Term_Seq]])))
, transpose(Array((data_hold[ data_hold.Term_Seq .== 3 , [:student_id, :Term_Seq]])))
, transpose(Array((data_hold[ data_hold.Term_Seq .== 4 , [:student_id, :Term_Seq]])))
, transpose(Array((data_hold[ data_hold.Term_Seq .== 5 , [:student_id, :Term_Seq]])))
, transpose(Array((data_hold[ data_hold.Term_Seq .== 6 , [:student_id, :Term_Seq]])))
)

id_flow_flat = transpose(hcat(id_flow[1],id_flow[2],id_flow[3],id_flow[4],id_flow[5],id_flow[6]))

output_flow_df = DataFrame(student_id = id_flow_flat[:,1], Term_Seq = id_flow_flat[:,2], θ_hat = θ_hat, y_hat= y_hat)

data_hold_w_pred = innerjoin(data_hold, output_flow_df, on = [:student_id, :Term_Seq])
maximum(data_hold_w_pred.θ_hat)

gdf = DataFrames.groupby(data_hold, :student_id)
outcome_df = combine(gdf, [:Outcome_23_Ind] .=> maximum; renamecols=true)

id_term_df = innerjoin(data_hold_w_pred, outcome_df, on = :student_id)

flow_df_1 = id_term_df[id_term_df.Outcome_23_Ind_maximum .==1, [:student_id, :Term_Seq, :θ_hat, :y_hat]]

path = "G:/My Drive/FSAN/9_Ret and Grad/Results/SNN_Flow_" * string(test_term_2) * ".csv"
CSV.write(path, flow_df_1)

### Average dropout probabilities from dropout and retained students
gdf = DataFrames.groupby(id_term_df[id_term_df.Outcome_23_Ind_maximum .==1, :], :Term_Seq)
outcome_df = combine(gdf, [:θ_hat] .=> mean; renamecols=true)
mean(id_term_df[id_term_df.Outcome_23_Ind_maximum .==1, :θ_hat])

gdf = DataFrames.groupby(id_term_df[id_term_df.Outcome_23_Ind_maximum .==0, :], :Term_Seq)
outcome_df = combine(gdf, [:θ_hat] .=> mean; renamecols=true)
mean(id_term_df[id_term_df.Outcome_23_Ind_maximum .==0, :θ_hat])


```
Plot the three integrations
```

Acads = []
Fins = []
Socios = []
loghs = []

for k in 1:n_model
    println("k=: ", k)
    model= NN_list[k]

    Acads_single = []
    Fins_single = []
    Socios_single = []
    logh_single = []

    for Seq in [2:2:6;]
        Acad_Score = sigmoid_fast.(model[Seq][1][1][2].bias .+ model[Seq][1][1][2].weight * sigmoid_fast.(model[Seq][1][1][1].bias .+ model[Seq][1][1][1].weight * x_test_surv[Seq][1]))
        Fin_Score = sigmoid_fast.(model[Seq][1][2][2].bias .+ model[Seq][1][2][2].weight * sigmoid_fast.(model[Seq][1][2][1].bias .+ model[Seq][1][2][1].weight * x_test_surv[Seq][2]))
        Socio_Score = sigmoid_fast.(model[Seq][1][3][2].bias .+ model[Seq][1][3][2].weight * sigmoid_fast.(model[Seq][1][3][1].bias .+ model[Seq][1][3][1].weight * x_test_surv[Seq][3]))   

        Acad_Integration = model[Seq][2].weight[1] * vec(Acad_Score)
        Fin_Integration = model[Seq][2].weight[2] * vec(Fin_Score)
        Socio_Integration = model[Seq][2].weight[3] * vec(Socio_Score)
        logh = vec(model[Seq][2].bias .+ model[Seq][2].weight * vcat(Acad_Score, Fin_Score, Socio_Score))   
        
        append!(Acads_single, Acad_Integration)
        append!(Fins_single, Fin_Integration)
        append!(Socios_single, Socio_Integration)
        append!(logh_single, logh)
    end

    push!(Acads, Acads_single)
    push!(Fins, Fins_single)
    push!(Socios, Socios_single)
    push!(loghs, logh_single)

end

Acad_matrix = hcat(Acads...)
Fin_matrix = hcat(Fins...)
Socio_matrix = hcat(Socios...)
logh_matrix = hcat(loghs...)

Acad_Integration = vec(mean(Acad_matrix, dims=2))
Fin_Integration = vec(mean(Fin_matrix, dims=2))
Socio_Integration = vec(mean(Socio_matrix, dims=2))
logh = vec(mean(logh_matrix, dims=2))

histogram(logh[y_test_acutal_flat.==0], xlabel="Log(hazard)", alpha=0.5, label="", normalize=true, xlims = (-7, 1)
#        , palette = :grays
        )
plot11 = histogram!(logh[y_test_acutal_flat.==1], xlabel="Log(hazard)", alpha=0.5, label="", normalize=true, xlims = (-7, 1)
#        , palette = :grays
        )
histogram(Acad_Integration[y_test_acutal_flat.==0], xlabel="Academic Integration", alpha=0.5, label="Retained", normalize=true, xlims = (-7, 1)
#        , palette = :grays
        )
plot21 = histogram!(Acad_Integration[y_test_acutal_flat.==1], xlabel="Academic Integration", alpha=0.5, label="Dropout", normalize=true, xlims = (-7, 1)
        )
histogram(Fin_Integration[y_test_acutal_flat.==0], xlabel="Economic Integration", alpha=0.5, label="", normalize=true, xlims = (-7, 1)
#        , palette = :grays
        )
plot31 = histogram!(Fin_Integration[y_test_acutal_flat.==1], xlabel="Economic Integration", alpha=0.5, label="", normalize=true, xlims = (-7, 1)
#        , palette = :grays
        )
histogram(Socio_Integration[y_test_acutal_flat.==0], xlabel="Social Integration", alpha=0.5, label="", normalize=true, xlims = (-7, 1)
#        , palette = :grays
        )
plot41 = histogram!(Socio_Integration[y_test_acutal_flat.==1], xlabel="Social Integration", alpha=0.5, label="", normalize=true, xlims = (-7, 1)
#        , palette = :grays
        )
plot(plot11, plot21, plot31, plot41, layout=(2,2),legend=true, size = (900, 700))

png("G:/My Drive/FSAN/9_Ret and Grad/Descr/Figure 2.png")
savefig("G:/My Drive/FSAN/9_Ret and Grad/Descr/Figure 2_20240116.pdf")