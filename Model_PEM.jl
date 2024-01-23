



```
Load packages
```

using Random, Distributions

using Plots, StatsPlots

using CSV, DataFrames

using FreqTables, StatsBase

using JuMP, Ipopt

using MultivariateStats

using MLUtils


```
Load data
```

raw = CSV.read("Ret_Surv_w_fold.csv", DataFrame)

names(raw)

```
CV Loop to find the best w
```  

for test_term in [2223:-10:2183;]
    train_term = test_term - 5
    data_train = raw[ (raw.academic_term .<test_term) .& (raw.Term_Seq .<=6) .& (raw.fold .!= 10), :]

    data_train.Period_length .= 1
    data_train.Outcome_23_Ind = data_train.Outcome_2_Ind .+ data_train.Outcome_3_Ind

    n_fit = nrow(data_train)

    ### Find the best w

    w_list = [1.1:0.1:1.9;]
    vldt_F_score = zeros(9)

    for fold in 1:9
        w = w_list[fold]

        data_vldt = data_train[ data_train.fold .!= fold, :]

        surv_fit = Model(Ipopt.Optimizer)

        n_Period = maximum(data_vldt.Term_Seq)
        l = -20 .* ones(n_Period)  
        u = 20 .* ones(n_Period)  
        @variable(surv_fit, l[i] <= β0_1[i = 1:n_Period] <= u[i])
        
        n_var = 29
        l_β = -20 .* ones(n_var)  
        u_β = 20 .* ones(n_var)  
        @variable(surv_fit, l_β[i] <= β_1[i = 1:n_Period, 1:n_var] <= u_β[i])       
        
        @NLexpression(
            surv_fit,
            Xβ_1[i = 1:n_fit], β0_1[data_vldt.Term_Seq[i]] 
                                + β_1[data_vldt.Term_Seq[i],1] * data_vldt.Pell_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],2] * data_vldt.Fst_Gen_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],3] * data_vldt.Gender_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],4] * data_vldt.Resid_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],5] * data_vldt.age[i]
                                + β_1[data_vldt.Term_Seq[i],6] * data_vldt.hs_gpa[i]
                                + β_1[data_vldt.Term_Seq[i],7] * data_vldt.Eth_ASIAN_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],8] * data_vldt.Eth_BLACK_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],9] * data_vldt.Eth_HISPA_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],10] * data_vldt.Eth_WHITE_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],11] * data_vldt.FT_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],12] * data_vldt.US_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],13] * data_vldt.D_Major_Ind[i]
                                + β_1[data_vldt.Term_Seq[i],14] * data_vldt.cum_gpa[i]
                                + β_1[data_vldt.Term_Seq[i],15] * data_vldt.DFW_Count_last[i]
                                + β_1[data_vldt.Term_Seq[i],16] * data_vldt.Listen_Count_last[i]
                                + β_1[data_vldt.Term_Seq[i],17] * data_vldt.Acad_stndng_Last[i]
                                + β_1[data_vldt.Term_Seq[i],18] * data_vldt.Tot_Credit_Rate[i]
                                + β_1[data_vldt.Term_Seq[i],19] * data_vldt.cur_regis_rate[i]
                                + β_1[data_vldt.Term_Seq[i],20] * data_vldt.Tot_Credit_MinorTerm_Rate[i]
                                + β_1[data_vldt.Term_Seq[i],21] * data_vldt.TOT_TRNSFR_rate[i]
                                + β_1[data_vldt.Term_Seq[i],22] * data_vldt.TOT_TEST_CREDIT_rate[i]
                                + β_1[data_vldt.Term_Seq[i],23] * data_vldt.fed_efc_rate[i]
                                + β_1[data_vldt.Term_Seq[i],24] * data_vldt.total_grant_rate[i]
                                + β_1[data_vldt.Term_Seq[i],25] * data_vldt.parent_loan_rate[i] 
                                + β_1[data_vldt.Term_Seq[i],26] * data_vldt.student_loan_rate[i] 
                                + β_1[data_vldt.Term_Seq[i],27] * data_vldt.work_study_rate[i] 
                                + β_1[data_vldt.Term_Seq[i],28] * data_vldt.Balance_Rate[i] 
                                + β_1[data_vldt.Term_Seq[i],29] * data_vldt.on_campus_ind[i] 
        
            )     
            
        @NLobjective(
            surv_fit,
                Min,
                -sum( w*data_vldt.Outcome_23_Ind[i]*Xβ_1[i] 
                     -exp(Xβ_1[i])*1
                    for i = 1:n_fit 
                    ) / n_fit
            )
                    
        optimize!(surv_fit)            

        β0_1_fit = value.(β0_1)
        β_1_fit = value.(β_1)

        # Validate the models        

        data_vldt = data_train[ data_train.fold .== fold, :]
        n_vldt = nrow(data_vldt)
        data_vldt.Outcome_23_Ind = data_vldt.Outcome_2_Ind .+ data_vldt.Outcome_3_Ind          
        
        Xβ_1_fit = (β0_1_fit[data_vldt.Term_Seq] 
        .+ β_1_fit[data_vldt.Term_Seq,1] .* data_vldt.Pell_Ind
        .+ β_1_fit[data_vldt.Term_Seq,2] .* data_vldt.Fst_Gen_Ind
        .+ β_1_fit[data_vldt.Term_Seq,3] .* data_vldt.Gender_Ind
        .+ β_1_fit[data_vldt.Term_Seq,4] .* data_vldt.Resid_Ind
        .+ β_1_fit[data_vldt.Term_Seq,5] .* data_vldt.age
        .+ β_1_fit[data_vldt.Term_Seq,6] .* data_vldt.hs_gpa
        .+ β_1_fit[data_vldt.Term_Seq,7] .* data_vldt.Eth_ASIAN_Ind
        .+ β_1_fit[data_vldt.Term_Seq,8] .* data_vldt.Eth_BLACK_Ind
        .+ β_1_fit[data_vldt.Term_Seq,9] .* data_vldt.Eth_HISPA_Ind
        .+ β_1_fit[data_vldt.Term_Seq,10] .* data_vldt.Eth_WHITE_Ind
        .+ β_1_fit[data_vldt.Term_Seq,11] .* data_vldt.FT_Ind
        .+ β_1_fit[data_vldt.Term_Seq,12] .* data_vldt.US_Ind
        .+ β_1_fit[data_vldt.Term_Seq,13] .* data_vldt.D_Major_Ind
        .+ β_1_fit[data_vldt.Term_Seq,14] .* data_vldt.cum_gpa
        .+ β_1_fit[data_vldt.Term_Seq,15] .* data_vldt.DFW_Count_last
        .+ β_1_fit[data_vldt.Term_Seq,16] .* data_vldt.Listen_Count_last
        .+ β_1_fit[data_vldt.Term_Seq,17] .* data_vldt.Acad_stndng_Last
        .+ β_1_fit[data_vldt.Term_Seq,18] .* data_vldt.Tot_Credit_Rate
        .+ β_1_fit[data_vldt.Term_Seq,19] .* data_vldt.cur_regis_rate
        .+ β_1_fit[data_vldt.Term_Seq,20] .* data_vldt.Tot_Credit_MinorTerm_Rate
        .+ β_1_fit[data_vldt.Term_Seq,21] .* data_vldt.TOT_TRNSFR_rate
        .+ β_1_fit[data_vldt.Term_Seq,22] .* data_vldt.TOT_TEST_CREDIT_rate
        .+ β_1_fit[data_vldt.Term_Seq,23] .* data_vldt.fed_efc_rate
        .+ β_1_fit[data_vldt.Term_Seq,24] .* data_vldt.total_grant_rate
        .+ β_1_fit[data_vldt.Term_Seq,25] .* data_vldt.parent_loan_rate
        .+ β_1_fit[data_vldt.Term_Seq,26] .* data_vldt.student_loan_rate
        .+ β_1_fit[data_vldt.Term_Seq,27] .* data_vldt.work_study_rate
        .+ β_1_fit[data_vldt.Term_Seq,28] .* data_vldt.Balance_Rate
        .+ β_1_fit[data_vldt.Term_Seq,29] .* data_vldt.on_campus_ind
        )
        
        λ_1_fit = exp.(Xβ_1_fit)
        S_1_fit = exp.(-λ_1_fit .* 1)
        θ_vldt_fit = 1 .- S_1_fit        
        
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

            vldt_conf_mtx = freqtable(data_vldt.Outcome_23_Ind, y_vldt_pred)
        
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

    ### Summarize CV results
    CV_results = DataFrame(Test_Term = test_term, w = w_select)
    path = "G:\\My Drive\\FSAN\\9_Ret and Grad\\Temp results"
    file = "PEM_CV_" * string(test_term) * ".csv"
    CSV.write(joinpath(path,file), CV_results)    
end

```
Model
Linear Xβ
```

for test_term_2 in [2223:-10:2203;] 

    ### Data Split
    fit_term = test_term_2 - 5
    data_fit = raw[ (raw.academic_term .<=fit_term) .& (raw.Term_Seq .<=6) .& (raw.fold .!= 10),:]
    n_fit = nrow(data_fit)
    data_fit.Period_length .= 1
    data_fit.Outcome_23_Ind = data_fit.Outcome_2_Ind .+ data_fit.Outcome_3_Ind

    ###
    data_test = raw[ (raw.academic_term .==test_term_2) .& (raw.Term_Seq .<=6), :]
    n_test = nrow(data_test)
    data_test.Outcome_23_Ind = data_test.Outcome_2_Ind .+ data_test.Outcome_3_Ind

    ### Model fitting
    surv_fit = Model(Ipopt.Optimizer)

    n_Period = maximum(data_fit.Term_Seq)
    l = -20 .* ones(n_Period)  
    u = 20 .* ones(n_Period)  
    @variable(surv_fit, l[i] <= β0_1[i = 1:n_Period] <= u[i])
    
    n_var = 29
    l_β = -20 .* ones(n_var)  
    u_β = 20 .* ones(n_var)  
    @variable(surv_fit, l_β[i] <= β_1[i = 1:n_Period, 1:n_var] <= u_β[i])
    
    @NLexpression(
        surv_fit,
        Xβ_1[i = 1:n_fit], β0_1[data_fit.Term_Seq[i]] 
                            + β_1[data_fit.Term_Seq[i],1] * data_fit.Pell_Ind[i]
                            + β_1[data_fit.Term_Seq[i],2] * data_fit.Fst_Gen_Ind[i]
                            + β_1[data_fit.Term_Seq[i],3] * data_fit.Gender_Ind[i]
                            + β_1[data_fit.Term_Seq[i],4] * data_fit.Resid_Ind[i]
                            + β_1[data_fit.Term_Seq[i],5] * data_fit.age[i]
                            + β_1[data_fit.Term_Seq[i],6] * data_fit.hs_gpa[i]
                            + β_1[data_fit.Term_Seq[i],7] * data_fit.Eth_ASIAN_Ind[i]
                            + β_1[data_fit.Term_Seq[i],8] * data_fit.Eth_BLACK_Ind[i]
                            + β_1[data_fit.Term_Seq[i],9] * data_fit.Eth_HISPA_Ind[i]
                            + β_1[data_fit.Term_Seq[i],10] * data_fit.Eth_WHITE_Ind[i]
                            + β_1[data_fit.Term_Seq[i],11] * data_fit.FT_Ind[i]
                            + β_1[data_fit.Term_Seq[i],12] * data_fit.US_Ind[i]
                            + β_1[data_fit.Term_Seq[i],13] * data_fit.D_Major_Ind[i]
                            + β_1[data_fit.Term_Seq[i],14] * data_fit.cum_gpa[i]
                            + β_1[data_fit.Term_Seq[i],15] * data_fit.DFW_Count_last[i]
                            + β_1[data_fit.Term_Seq[i],16] * data_fit.Listen_Count_last[i]
                            + β_1[data_fit.Term_Seq[i],17] * data_fit.Acad_stndng_Last[i]
                            + β_1[data_fit.Term_Seq[i],18] * data_fit.Tot_Credit_Rate[i]
                            + β_1[data_fit.Term_Seq[i],19] * data_fit.cur_regis_rate[i]
                            + β_1[data_fit.Term_Seq[i],20] * data_fit.Tot_Credit_MinorTerm_Rate[i]
                            + β_1[data_fit.Term_Seq[i],21] * data_fit.TOT_TRNSFR_rate[i]
                            + β_1[data_fit.Term_Seq[i],22] * data_fit.TOT_TEST_CREDIT_rate[i]
                            + β_1[data_fit.Term_Seq[i],23] * data_fit.fed_efc_rate[i]
                            + β_1[data_fit.Term_Seq[i],24] * data_fit.total_grant_rate[i]
                            + β_1[data_fit.Term_Seq[i],25] * data_fit.parent_loan_rate[i] 
                            + β_1[data_fit.Term_Seq[i],26] * data_fit.student_loan_rate[i] 
                            + β_1[data_fit.Term_Seq[i],27] * data_fit.work_study_rate[i] 
                            + β_1[data_fit.Term_Seq[i],28] * data_fit.Balance_Rate[i] 
                            + β_1[data_fit.Term_Seq[i],29] * data_fit.on_campus_ind[i] 
    
        )
    
    w_1 = 1.4

    @NLobjective(
        surv_fit,
            Min,
            -sum( w_1*data_fit.Outcome_23_Ind[i]*Xβ_1[i] 
                 -exp(Xβ_1[i])*1
                for i = 1:n_fit 
                ) / n_fit
        )
    
    optimize!(surv_fit)    

    ### Derive predicted dropout probabilities in the training dataset
    β0_1_fit = value.(β0_1)
    β_1_fit = value.(β_1)
    
    Xβ_1_fit = (β0_1_fit[data_fit.Term_Seq] 
    .+ β_1_fit[data_fit.Term_Seq,1] .* data_fit.Pell_Ind
    .+ β_1_fit[data_fit.Term_Seq,2] .* data_fit.Fst_Gen_Ind
    .+ β_1_fit[data_fit.Term_Seq,3] .* data_fit.Gender_Ind
    .+ β_1_fit[data_fit.Term_Seq,4] .* data_fit.Resid_Ind
    .+ β_1_fit[data_fit.Term_Seq,5] .* data_fit.age
    .+ β_1_fit[data_fit.Term_Seq,6] .* data_fit.hs_gpa
    .+ β_1_fit[data_fit.Term_Seq,7] .* data_fit.Eth_ASIAN_Ind
    .+ β_1_fit[data_fit.Term_Seq,8] .* data_fit.Eth_BLACK_Ind
    .+ β_1_fit[data_fit.Term_Seq,9] .* data_fit.Eth_HISPA_Ind
    .+ β_1_fit[data_fit.Term_Seq,10] .* data_fit.Eth_WHITE_Ind
    .+ β_1_fit[data_fit.Term_Seq,11] .* data_fit.FT_Ind
    .+ β_1_fit[data_fit.Term_Seq,12] .* data_fit.US_Ind
    .+ β_1_fit[data_fit.Term_Seq,13] .* data_fit.D_Major_Ind
    .+ β_1_fit[data_fit.Term_Seq,14] .* data_fit.cum_gpa
    .+ β_1_fit[data_fit.Term_Seq,15] .* data_fit.DFW_Count_last
    .+ β_1_fit[data_fit.Term_Seq,16] .* data_fit.Listen_Count_last
    .+ β_1_fit[data_fit.Term_Seq,17] .* data_fit.Acad_stndng_Last
    .+ β_1_fit[data_fit.Term_Seq,18] .* data_fit.Tot_Credit_Rate
    .+ β_1_fit[data_fit.Term_Seq,19] .* data_fit.cur_regis_rate
    .+ β_1_fit[data_fit.Term_Seq,20] .* data_fit.Tot_Credit_MinorTerm_Rate
    .+ β_1_fit[data_fit.Term_Seq,21] .* data_fit.TOT_TRNSFR_rate
    .+ β_1_fit[data_fit.Term_Seq,22] .* data_fit.TOT_TEST_CREDIT_rate
    .+ β_1_fit[data_fit.Term_Seq,23] .* data_fit.fed_efc_rate
    .+ β_1_fit[data_fit.Term_Seq,24] .* data_fit.total_grant_rate
    .+ β_1_fit[data_fit.Term_Seq,25] .* data_fit.parent_loan_rate
    .+ β_1_fit[data_fit.Term_Seq,26] .* data_fit.student_loan_rate
    .+ β_1_fit[data_fit.Term_Seq,27] .* data_fit.work_study_rate
    .+ β_1_fit[data_fit.Term_Seq,28] .* data_fit.Balance_Rate
    .+ β_1_fit[data_fit.Term_Seq,29] .* data_fit.on_campus_ind
    )
    
    
    λ_1_fit = exp.(Xβ_1_fit)
    S_1_fit = exp.(-λ_1_fit .* 1)
    data_fit.θ_1_raw = 1 .- S_1_fit

    ### Find threshold probabilities
    thres_slct = []
    thres_list = [0.15:0.01:0.50;]
    
    for j in 1:6
        θ_pred = data_fit.θ_1_raw[data_fit.Term_Seq .== j]
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
    
            fit_conf_mtx = freqtable(data_fit.Outcome_23_Ind[data_fit.Term_Seq .== j], y_pred)
            fit_precise = fit_conf_mtx[4]/(fit_conf_mtx[4]+fit_conf_mtx[3])
            fit_recall = fit_conf_mtx[4]/(fit_conf_mtx[4]+fit_conf_mtx[2])
            fit_F_score = 2*fit_precise*fit_recall/(fit_precise+fit_recall)
            push!(F_scores, fit_F_score)
    
        end
    
        push!(thres_slct,thres_list[findmax(F_scores)[2]])
    end
    
    push!(thres_slcts , thres_slct)

    ### Derive F1-score for the test dataset

    Xβ_1_test = (β_1_fit[data_test.Term_Seq] 
    .+ β_1_fit[data_test.Term_Seq,1] .* data_test.Pell_Ind
    .+ β_1_fit[data_test.Term_Seq,2] .* data_test.Fst_Gen_Ind
    .+ β_1_fit[data_test.Term_Seq,3] .* data_test.Gender_Ind
    .+ β_1_fit[data_test.Term_Seq,4] .* data_test.Resid_Ind
    .+ β_1_fit[data_test.Term_Seq,5] .* data_test.age
    .+ β_1_fit[data_test.Term_Seq,6] .* data_test.hs_gpa
    .+ β_1_fit[data_test.Term_Seq,7] .* data_test.Eth_ASIAN_Ind
    .+ β_1_fit[data_test.Term_Seq,8] .* data_test.Eth_BLACK_Ind
    .+ β_1_fit[data_test.Term_Seq,9] .* data_test.Eth_HISPA_Ind
    .+ β_1_fit[data_test.Term_Seq,10] .* data_test.Eth_WHITE_Ind
    .+ β_1_fit[data_test.Term_Seq,11] .* data_test.FT_Ind
    .+ β_1_fit[data_test.Term_Seq,12] .* data_test.US_Ind
    .+ β_1_fit[data_test.Term_Seq,13] .* data_test.D_Major_Ind
    .+ β_1_fit[data_test.Term_Seq,14] .* data_test.cum_gpa
    .+ β_1_fit[data_test.Term_Seq,15] .* data_test.DFW_Count_last
    .+ β_1_fit[data_test.Term_Seq,16] .* data_test.Listen_Count_last
    .+ β_1_fit[data_test.Term_Seq,17] .* data_test.Acad_stndng_Last
    .+ β_1_fit[data_test.Term_Seq,18] .* data_test.Tot_Credit_Rate
    .+ β_1_fit[data_test.Term_Seq,19] .* data_test.cur_regis_rate
    .+ β_1_fit[data_test.Term_Seq,20] .* data_test.Tot_Credit_MinorTerm_Rate
    .+ β_1_fit[data_test.Term_Seq,21] .* data_test.TOT_TRNSFR_rate
    .+ β_1_fit[data_test.Term_Seq,22] .* data_test.TOT_TEST_CREDIT_rate
    .+ β_1_fit[data_test.Term_Seq,23] .* data_test.fed_efc_rate
    .+ β_1_fit[data_test.Term_Seq,24] .* data_test.total_grant_rate
    .+ β_1_fit[data_test.Term_Seq,25] .* data_test.parent_loan_rate
    .+ β_1_fit[data_test.Term_Seq,26] .* data_test.student_loan_rate
    .+ β_1_fit[data_test.Term_Seq,27] .* data_test.work_study_rate
    .+ β_1_fit[data_test.Term_Seq,28] .* data_test.Balance_Rate
    .+ β_1_fit[data_test.Term_Seq,29] .* data_test.on_campus_ind
    )

    λ_1_test = exp.(Xβ_1_test)
    S_1_test = exp.(-λ_1_test .* 1)
    data_test.θ_1_raw = 1 .- S_1_test

    θ_test_order = [data_test.θ_1_raw[data_test.Term_Seq .== 2]; data_test.θ_1_raw[data_test.Term_Seq .== 4];data_test.θ_1_raw[data_test.Term_Seq .== 6]]

    thres_slct_list = vcat(ones(sum(data_test.Term_Seq .== 2))*thres_slct[2], ones(sum(data_test.Term_Seq .== 4))*thres_slct[4], ones(sum(data_test.Term_Seq .== 6))*thres_slct[6])
    y_test_pred = θ_test_order .>= thres_slct_list
    y_test_order = [data_test.Outcome_23_Ind[data_test.Term_Seq .== 2]; data_test.Outcome_23_Ind[data_test.Term_Seq .== 4]; data_test.Outcome_23_Ind[data_test.Term_Seq .== 6]]

    test_conf_mtx = freqtable(y_test_order, y_test_pred)
    test_precise = test_conf_mtx[4]/(test_conf_mtx[4]+test_conf_mtx[3])
    test_recall = test_conf_mtx[4]/(test_conf_mtx[4]+test_conf_mtx[2])
    test_F_score = 2*test_precise*test_recall/(test_precise+test_recall)
    display(test_conf_mtx)
    println( "test_precise= ",test_precise, " test_recall= ", test_recall, " test_F_score= ", test_F_score)

end