import random
import torch
import numpy as np
import torch.nn as nn
import math
from tqdm import tqdm 
from itertools import combinations
from client_fed_local import federation
from utils import transform_dict_to_lists, shapley,combine_plot,plot_coeficits, cosine_similarity_models
from utils import approx_quantities,pri_ce, mean_columns_list
from loggers import logger_f#metricslogger
from utils import plot_boxplot_with_stats
import os
import pathlib
from modelos import CNN, CNN_brain, MLP, LogisticRegressionModel
import argparse
from local_breast import commun_test_set 
import matplotlib.pyplot as plt




torch.cuda.is_available()
# dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)
#dev = "cpu"
#This class contain the functions to be run. The main function to be consider is the one call statistic, with the 
#output of this function with can compute the mean and std of each ppce with respect to the sv and the values of 
#the ppce and sv of each client at the last iteration of fed and the sv anf ppce accumulative over rounds


class games(federation):
    def __init__(self, model_type, dataset_name, num_clients,partition_type,alpha,device=dev):
        super().__init__(model_type, dataset_name, num_clients,partition_type,alpha,device=dev)
    
    
    def retrainin_game(self,iterations, con_clientes, epochs=5):
        if len(con_clientes) == 0:
            return 0
        self.reset_parameters()
        self.federated_averaging(iterations,con_clientes,num_epochs=epochs)
        worth = self.evaluation_global(self.test_global)
        return worth
    
    
    def coalitional_values(self,iteration,path,epochs=5):
        d = dict()
        d[()]=0
        for client in range(self.num_clients):
            self.lista_clientes[client].fit(epochs)
            d[tuple([client])] = self.lista_clientes[client].evaluation(self.test_global)
            self.lista_clientes[client].reset_parameters()
        for subset_len in range(2,self.num_clients+1):
            for subset_idx in combinations(list(range(self.num_clients)), subset_len):
                d[subset_idx] = self.retrainin_game(iteration,list(subset_idx),epochs)
            tqdm.write(f"subset size {subset_len}")
        logger_f(f"True coalitional values retraining: {d}",f"{path}/values/results_retraining.log")
        return d
    
    def traing_together(self, num_epochs):
        """sort of function that does  fedAvg"""
        for client in self.lista_clientes:
            weights_modelo = self.model.state_dict()
            client.load_state_dict(weights_modelo)
            client.fit(num_epochs)
        params = self.weighted_aggre(self.lista_clientes)
        self.set_parameters(params)
       

    
    def local_coalicional_value_at_round(self,iter,path,epochs=5):
        """
        iter:iteracion
        output: dictionary. keys are tuples that represent each coalition, values are the evaluation of the model of each coalition.  
        """
        for i in range(iter+1):
            self.traing_together(epochs)
            tqdm.write(f"iteration_fed={i}")
        d_privacy= dict()
        d_privacy[()]=0
        for client in range(self.num_clients):
            remove=[j for j in range(self.num_clients) if j!=client]
            d_privacy[tuple([client])] = (self.evaluation_global(self.lista_clientes[client].test),self.lista_clientes[client].evaluation(self.lista_clientes[client].test))
            part=[self.lista_clientes[i] for i in remove]
            params = self.weighted_aggre(part)
            self.set_parameters(params)
            d_privacy[tuple(remove)] = (self.evaluation_global(self.lista_clientes[client].test))
        logger_f(f"at coalitional values: {d_privacy}",f"{path}/values/results_local.log")
        d = dict()
        d[()]=0
        for client in range(self.num_clients):
            d[tuple([client])] = self.lista_clientes[client].evaluation(self.test_global)
        for subset_len in range(2,self.num_clients+1):
            for subset_idx in combinations(list(range(self.num_clients)), subset_len):
                part=[self.lista_clientes[i] for i in subset_idx]
                params = self.weighted_aggre(part)
                self.set_parameters(params)
                d[subset_idx] = self.evaluation_global(self.test_global)
            tqdm.write(f"subset size {subset_len}")
        logger_f(f"at coalitional values: {d}",f"{path}/values/results_global.log")
        return d_privacy,d
    

    def cosine_sim_clie_server(self):
        co_si=[]
        for client in self.lista_clientes:
            value=cosine_similarity_models(client.model,self.model)
            co_si.append(value)
        logger_f(f"Cosine: {co_si}","cosine")
        return co_si
    
    def local_pric_ce(self,dict,path):
        i1i = np.zeros(self.num_clients)
        l1o = np.zeros(self.num_clients)
        ieei = np.zeros(self.num_clients)
        leeo = np.zeros(self.num_clients)
        include = np.zeros(self.num_clients)
        leave = np.zeros(self.num_clients)
        grands = np.zeros(self.num_clients)
        null=0
        for i in range(self.num_clients):
            remove=[j for j in range(self.num_clients) if j!=i]
            for tpl in dict.keys():
                if tpl == tuple([i]):
                    include[i] = dict[tpl][1]
                    grands[i] =dict[tpl][0]
                if tpl == tuple(remove):
                    leave[i] = dict[tpl]
        for i in range(self.num_clients):
            i1i[i] = (include[i] - null)
            l1o[i] = -(grands[i] - leave[i])
            for j in range(self.num_clients):
                ieei[i] += (leave[j] - null)
                leeo[i] += (grands[j] - include[j])
            ieei[i] -= (leave[i] - null)
            leeo[i] -= (grands[i] - include[i])
        ieei= -ieei / (self.num_clients - 1) ** 2
        leeo= -leeo / (self.num_clients - 1) ** 2
        se = (i1i + l1o)/2
        ee = (ieei + leeo)/2
        ppce = (se + ee)/2
        logger_f(f"i1i: {i1i.tolist()}",f"{path}/values/ppce_local.log")
        logger_f(f"l10: {l1o.tolist()}",f"{path}/values/ppce_local.log")
        logger_f(f"ie2i: {ieei.tolist()}",f"{path}/values/ppce_local.log")
        logger_f(f"le2o: {leeo.tolist()}",f"{path}/values/ppce_local.log")
        logger_f(f"se: {se.tolist()}",f"{path}/values/ppce_local.log")
        logger_f(f"ee: {ee.tolist()}",f"{path}/values/ppce_local.log")
        logger_f(f"PPce: {ppce.tolist()}",f"{path}/values/ppce_local.log")
        return [i1i.tolist(), l1o.tolist(), ieei.tolist(), leeo.tolist(),se.tolist(),ee.tolist(),ppce.tolist()]
    
    def coalition_plot(self,d_local, d_global):
        tpl_global=tuple(range(self.num_clients))
        for i in range(self.num_clients):
            ieei_local = np.zeros(self.num_clients)
            ieei_global = np.zeros(self.num_clients)
            leeo_local = np.zeros(self.num_clients)
            leeo_global = np.zeros(self.num_clients)
            for j in range(self.num_clients):
                if j!=i:
                    remove=[k for k in range(self.num_clients) if k!=j]
                    ieei_local[j]= d_local[tuple(remove)]            
                    ieei_global[j]= d_global[tuple(remove)]            
                    leeo_global[j]= d_global[tpl_global]-d_global[tuple([j])]            
                    leeo_local[j]= d_local[tuple([j])][0]-d_local[tuple([j])][1]            
                else:
                    ieei_local[j]= 0           
                    ieei_global[j]= 0           
                    leeo_global[j]= 0           
                    leeo_local[j]= 0
            x_labels = [f"Coa. {j}" for j in range(self.num_clients)]
            x = np.arange(len(x_labels))  # the label locations
            # Plot for ieei_local
            plt.figure(figsize=(10, 6))
            plt.bar(x, ieei_local, color='blue', width=0.2,alpha=0.7, label='ieei_local')
            plt.bar(x+0.2, ieei_global, color='green', width=0.2 , alpha=0.7, label='ieei_global')
            plt.xticks(x, x_labels, rotation=45)
            plt.xlabel("Clients")
            plt.ylabel("Values")
            plt.title(f"IEEI Local for Client {i}")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for readability
            plt.tight_layout()
            plt.savefig(f"./RESULTS_local/ieei_local_global_client_{i}.png")  # Save the plot as a PNG file
            plt.close()

            # Plot for ieei_global
            # plt.figure(figsize=(10, 6))
            # plt.xticks(x, x_labels, rotation=45)
            # plt.xlabel("Clients")
            # plt.ylabel("Values")
            # plt.title(f"IEEI Global for Client {i}")
            # plt.legend()
            # plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for readability
            # plt.tight_layout()
            # plt.savefig(f"./RESULTS_local/ieei_global_client_{i}.png")  # Save the plot as a PNG file
            # plt.close()
            
    
        

    def valores_por_rondas(self,itera,path,epochs=5):   
            """
            itera: number of federation iterations 
            output: two lists. The first one contains the sv and ppce for each client at the last round. 
            The second list contains the sv and ppce cumulative over the all itera (cummulative means the mean)
            """
            for client in range(self.num_clients):
              self.lista_clientes[client].reset_parameters()
            self.reset_parameters()
            dic_values=self.local_coalicional_value_at_round(itera,path,epochs=epochs)  
            local_pri=self.local_pric_ce(dic_values[0],path) 
            trans=transform_dict_to_lists(dic_values[1],self.num_clients)
            priv_pr=pri_ce(self.num_clients,trans[0],trans[1],path)
            base_metr=shapley(self.num_clients,trans[0],trans[1],path)
            cosine=self.cosine_sim_clie_server()
            # efecto=effect(self.num_clients,trans[0],trans[1])
            return base_metr,local_pri,priv_pr,cosine#efecto

####################################comparation sv,svcosine, privacy preserving

    def cosine_coalicional_value(self,iter,path,epochs=5):
            """
            iter:iteracion
            output: dictionary. keys are tuples that represent each coalition, values are the evaluation of the model of each coalition.  
            """
            for i in range(iter+1):
                self.traing_together(epochs)
                tqdm.write(f"iteration_fed={i}")
            dummy_model=CNN_brain()#LogisticRegressionModel()
            dummy_model.set_parameters(self.parametros())
            d_cosim= dict()
            d_cosim[()]=0
            for client in range(self.num_clients):
                d_cosim[tuple([client])] = cosine_similarity_models(self.lista_clientes[client],dummy_model)
            for subset_len in range(2,self.num_clients+1):
                for subset_idx in combinations(list(range(self.num_clients)), subset_len):
                    part_cos=[self.lista_clientes[i] for i in subset_idx]
                    params_cos = self.weighted_aggre(part_cos)
                    self.set_parameters(params_cos)
                    d_cosim[subset_idx] = cosine_similarity_models(self.model,dummy_model)
                tqdm.write(f"subset size {subset_len}")
            logger_f(f"at coalitional values_cosine: {d_cosim}",f"{path}/values/results_cosine.log")
            d = dict()
            d[()]=0
            for client in range(self.num_clients):
                d[tuple([client])] = self.lista_clientes[client].evaluation(self.test_global)
            for subset_len in range(2,self.num_clients+1):
                for subset_idx in combinations(list(range(self.num_clients)), subset_len):
                    part=[self.lista_clientes[i] for i in subset_idx]
                    params = self.weighted_aggre(part)
                    self.set_parameters(params)
                    d[subset_idx] = self.evaluation_global(self.test_global)
                tqdm.write(f"subset size {subset_len}")
            logger_f(f"at coalitional values: {d}",f"{path}/values/results_evaluation.log")
            return d_cosim,d


    def val_por_ron_cosine(self,itera,path, epochs):
        for client in range(self.num_clients):
            self.lista_clientes[client].reset_parameters()
        self.reset_parameters()
        dic_values=self.cosine_coalicional_value(itera,path,epochs=epochs)  
        trans1=transform_dict_to_lists(dic_values[0],self.num_clients)
        trans2=transform_dict_to_lists(dic_values[1],self.num_clients)
        priv_pr=pri_ce(self.num_clients,trans2[0],trans2[1],path)
        normal_sv=shapley(self.num_clients,trans2[0],trans2[1],path)
        cosine_sv=shapley(self.num_clients,trans1[0],trans1[1],path)
        return normal_sv, priv_pr, cosine_sv
    
    def statistic_cosine(self,path,iter_one,iter_two=10,epochs=5):
        """
        iter_one:number of iteration to run the federation several times 
        iter_two:number of federation iterations
        output1: matrix with each row the  
        """
        ####global
        rat_err_lr=[]
        spear_lr=[]
        pear_err_lr=[]
        kend_lr=[]
        #matrix_of_scores_values
        matrix_scores_last_rounds_gl=[]
        for i in range(iter_one):
            valores=self.val_por_ron_cosine(iter_two,path,epochs)
            #matrix
            for_mean_glo=[valores[0]]+valores[1]
            matrix_scores_last_rounds_gl.append(np.array(for_mean_glo).T)
            #global
            valor_wc=valores[1]+[valores[2]]
            metrlr=approx_quantities(valores[0],valor_wc,f"{path}/values/metrics_global_cosine_sv.log")
            rat_err_lr.append([metrlr[j][0] for j in range(len(valor_wc))])
            spear_lr.append([metrlr[k][1] for k in range(len(valor_wc))])
            pear_err_lr.append([metrlr[j][2] for j in range(len(valor_wc))])
            kend_lr.append([metrlr[k][3] for k in range(len(valor_wc))])
            #local
            tqdm.write(f"iteration_global={i}")
            #"global":
        lse_dir_cos=np.array(rat_err_lr)[:, [-1]]-np.array(rat_err_lr)[:, :-1]
        spear_dir_cos=np.array(spear_lr)[:, :-1]-np.array(spear_lr)[:, [-1]]
        pear_dir_cos= np.array(pear_err_lr)[:, :-1]- np.array(pear_err_lr)[:, [-1]]
        kend_dir_cos=np.array(kend_lr)[:, :-1]-np.array(kend_lr)[:, [-1]]
        arrays_lr=[np.array(rat_err_lr)[:, :-1],np.array(spear_lr)[:, :-1],np.array(pear_err_lr)[:, :-1],np.array(kend_lr)[:, :-1],lse_dir_cos,spear_dir_cos,pear_dir_cos,kend_dir_cos]
        metrica=metrlr[:-1]
        return arrays_lr,valores,metrica,np.array(matrix_scores_last_rounds_gl)


    def simulation_cosine_sv(self,iteraci=10,iter_fed=10,epochs=1):
            if self.partition_type=="N-IID":
                path1=f'./RESULTS_cosine_sv/{self.dataset_name}/{self.num_clients}cli({self.alpha})_{iter_fed}fed'
                title=f"Case N-IID, alpha={self.alpha}, and iter={iter_fed}"
                os.makedirs(path1, exist_ok=True)
            elif self.partition_type=='IID':
                path1=f'./RESULTS_cosine_sv/{self.dataset_name}/{self.num_clients}cli({self.partition_type})_{iter_fed}fed'
                title=f"Case {self.partition_type}, and iter={iter_fed}"
                os.makedirs(path1, exist_ok=True)
            name0=f"values_clients_global"
            name2=f"correlations_global_test_set"
            metricas_global=["Normalize_lse_global_test_set","Spearman_global_test_set","Pearson_global_test_set","Kendall_global_test_set","Difer_Nor_lse_CoSim","Dirf_Spearman_CoSim","Dirf_Pearson_CoSim","Dirf_Kendall_CoSim"]
            transform_re=self.statistic_cosine(path1,iteraci,iter_fed,epochs=epochs)
            last_round_global=mean_columns_list(transform_re[3])#[transform_re[1][0]]+transform_re[1][2]
            combine_plot(last_round_global,["SV","I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name0)
            last_round_metr_global=transform_re[2]
            plot_coeficits(last_round_metr_global,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name2)
            self.boxplot(transform_re[0],metricas_global,path1,"last_round")
            return transform_re[0]#, transform_re[3]


    
##################################################

#this function has been modify to only consider the cos similarity. the original statistic function is test.py
    
    def statistic(self,path,iter_one,iter_two=10,epochs=5):
        """
        iter_one:number of iteration to run the federation several times 
        iter_two:number of federation iterations
        output1: matrix with each row the  
        """
        ####global
        rat_err_lr=[]
        spear_lr=[]
        pear_err_lr=[]
        kend_lr=[]
        #local
        rat_err_lr_local=[]
        spear_lr_local=[]
        pear_err_lr_local=[]
        kend_lr_local=[]
        #matrix_of_scores_values
        matrix_scores_last_rounds_gl=[]
        matrix_scores_last_rounds_lo=[]
        for i in range(iter_one):
            valores=self.valores_por_rondas(iter_two,path,epochs)
            #matrix
            for_mean_glo=[valores[0]]+valores[2]+[valores[3]]
            matrix_scores_last_rounds_gl.append(np.array(for_mean_glo).T)
            for_mean_lo=[valores[0]]+valores[1]
            matrix_scores_last_rounds_lo.append(np.array(for_mean_lo).T)
            #global
            valor_wc=valores[2]+[valores[3]]
            metrlr=approx_quantities(valores[0],valor_wc,f"{path}/values/metrics_global.log")
            rat_err_lr.append([metrlr[j][0] for j in range(len(valor_wc))])
            spear_lr.append([metrlr[k][1] for k in range(len(valor_wc))])
            pear_err_lr.append([metrlr[j][2] for j in range(len(valor_wc))])
            kend_lr.append([metrlr[k][3] for k in range(len(valor_wc))])
            #local
            valor_local=valores[1]
            metrlr_local=approx_quantities(valores[0],valor_local,f"{path}/values/metrics_local.log")
            rat_err_lr_local.append([metrlr_local[j][0] for j in range(len(valor_local))])
            spear_lr_local.append([metrlr_local[k][1] for k in range(len(valor_local))])
            pear_err_lr_local.append([metrlr_local[j][2] for j in range(len(valor_local))])
            kend_lr_local.append([metrlr_local[k][3] for k in range(len(valor_local))])
            tqdm.write(f"iteration_global={i}")
            #"global":
        #lse_dir_cos=np.array(rat_err_lr)[:, [-1]]-np.array(rat_err_lr)[:, :-1]
        #spear_dir_cos=np.array(spear_lr)[:, :-1]-np.array(spear_lr)[:, [-1]]
        #pear_dir_cos= np.array(pear_err_lr)[:, :-1]- np.array(pear_err_lr)[:, [-1]]
        #kend_dir_cos=np.array(kend_lr)[:, :-1]-np.array(kend_lr)[:, [-1]]
        arrays_lr=[np.array(rat_err_lr),np.array(spear_lr),np.array(pear_err_lr),np.array(kend_lr)]#,lse_dir_cos,spear_dir_cos,pear_dir_cos,kend_dir_cos]
        metrica=metrlr
            #"local":
        arrays_lr_local=[np.array(rat_err_lr_local),np.array(spear_lr_local),np.array(pear_err_lr_local),np.array(kend_lr_local)]
        metrica_local=metrlr_local
        return [arrays_lr,arrays_lr_local],valores,[metrica,metrica_local],np.array(matrix_scores_last_rounds_gl),np.array(matrix_scores_last_rounds_lo)#,#,metrmean,effects_EE
    

    def boxplot(self,valores,metricas,path,name,metrics_names,retraining=None):
        # if level == "global":
        #     metricas_global=["Normalize_lse","Spearman","Pearson","Kendall","Normalize_lse_CoSim","Spearman_CoSim","Pearson_CoSim","Kendall_CoSim"]
        # elif level== "local":
        #     metricas_local=["Normalize_lse","Spearman","Pearson","Kendall"]
        # valores=self.valores_por_rondas(ite)
        for i in range(len(valores)):
            length=len(valores[i][0])
            metric=metricas[i]
            avg_i1i=[row[0] for row in valores[i]]
            avg_l1o=[row[1] for row in valores[i]]
            avg_ieei=[row[2] for row in valores[i]]
            avg_leeo=[row[3] for row in valores[i]]
            avg_se=[row[4] for row in valores[i]]
            avg_ee=[row[5] for row in valores[i]]
            avg_ppce=[row[6] for row in valores[i]]
            if length>7:
                avg_fi1i=[row[7] for row in valores[i]]
                avg_fl1o=[row[8] for row in valores[i]]
                avg_cos=[row[9] for row in valores[i]]
                if retraining==None:
                    big_list=[avg_i1i,avg_l1o,avg_ieei,avg_leeo,avg_se,avg_ee,avg_ppce,avg_fi1i,avg_fl1o,avg_cos]
                else:
                    av_sv=[row[9] for row in valores[i]]
                    big_list=[avg_i1i,avg_l1o,avg_ieei,avg_leeo,avg_se,avg_ee,avg_ppce,avg_fi1i,avg_fl1o,av_sv]
            else:
                if retraining==None:
                    big_list=[avg_i1i,avg_l1o,avg_ieei,avg_leeo,avg_se,avg_ee,avg_ppce]
                else:
                    av_sv=[row[7] for row in valores[i]]
                    big_list=[avg_i1i,avg_l1o,avg_ieei,avg_leeo,avg_se,avg_ee,avg_ppce,av_sv]
            means = [np.mean(lst).item() for lst in big_list]
            stds = [np.std(lst).item() for lst in big_list]
            logger_f(f"means for {metric}: {means}",f"{path}/values/mean_std.log")
            logger_f(f"stds for {metric}: {stds}",f"{path}/values/mean_std.log")
            if retraining==None:
                plot_boxplot_with_stats(big_list,metrics_names,path,f"{metric}: {name}",f"{metric}: {name}")
            else: 
                plot_boxplot_with_stats(big_list,metrics_names+["Data_SV"],path,f"{metric}: {name}",f"{metric}: {name}")
        logger_f(f"{name}, {self.dataset_name}, {self.num_clients}, {self.alpha}","mean_std")
    
    def simulation_global_iter(self,iteraci=10,iter_fed=10,epochs=1):
            if self.partition_type=="N-IID":
                path1=f'./RESULTS_local/{self.dataset_name}/{self.num_clients}cli({self.alpha})_{iter_fed}fed'
                title=f"Case N-IID, alpha={self.alpha}, and iter={iter_fed}"
                os.makedirs(path1, exist_ok=True)
            elif self.partition_type=='IID':
                path1=f'./RESULTS_local/{self.dataset_name}/{self.num_clients}cli({self.partition_type})_{iter_fed}fed'
                title=f"Case {self.partition_type}, and iter={iter_fed}"
                os.makedirs(path1, exist_ok=True)
            name0=f"values_clients_global"
            name1=f"values_clients_local"
            name2=f"correlations_global_test_set"
            name3=f"correlations_local_test_set"
            metricas_global=["Normalize_lse_global_test_set","Spearman_global_test_set","Pearson_global_test_set","Kendall_global_test_set"]#"Difer_Nor_lse_CoSim","Dirf_Spearman_CoSim","Dirf_Pearson_CoSim","Dirf_Kendall_CoSim"]
            metricas_local=["Normalize_lse_local_test_set","Spearman_local_test_set","Pearson_local_test_set","Kendall_local_test_set"]
            transform_re=self.statistic(path1,iteraci,iter_fed,epochs=epochs)
            last_round_global=mean_columns_list(transform_re[3])#[transform_re[1][0]]+transform_re[1][2]
            last_round_local=mean_columns_list(transform_re[4])#[transform_re[1][0]]+transform_re[1][1]
            combine_plot(last_round_global,["SV","I1I","L1O","IE2I","LE2O","SE","EE","PPCE","f_I1I","f_L1O", "COS"],path1,title,name0)
            combine_plot(last_round_local,["SV","I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name1)
            last_round_metr_global=transform_re[2][0]
            last_round_metr_local=transform_re[2][1]
            plot_coeficits(last_round_metr_global,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE","f_I1I","f_L1O", "COS"],path1,title,name2)
            plot_coeficits(last_round_metr_local,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name3)
            self.boxplot(transform_re[0][0],metricas_global,path1,"last_round",["I1I","L1O","IE2I","LE2O","SE","EE","PPCE","f_I1I","f_L1O", "COS"])
            self.boxplot(transform_re[0][1],metricas_local,path1,"last_round",["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"])
            return transform_re[0], transform_re[3]
    
    def statistic_to_retr_game(self,path,iter_one,iter_two=10,epochs=5):
        """
        iter_one:number of iteration to run the federation several times 
        iter_two:number of federation iterations
        output1: matrix with each row the  
        """
        ####global
        rat_err_lr=[]
        spear_lr=[]
        pear_err_lr=[]
        kend_lr=[]
        #matrix_of_scores_values
        matrix_scores_last_rounds_re=[]
        true_values=self.coalitional_values(iter_two,path,epochs=epochs)  
        trans=transform_dict_to_lists(true_values,self.num_clients)
        true_sv=shapley(self.num_clients,trans[0],trans[1],path,retraining=1)
        for i in range(iter_one):
            valores=self.valores_por_rondas(iter_two,path,epochs)
            #global
            valor_wc=valores[2]+[valores[0]]
            matrix_scores_last_rounds_re.append(np.array(valor_wc).T)
            metrlr=approx_quantities(true_sv,valor_wc,f"{path}/values/metrics_retraining.log")
            rat_err_lr.append([metrlr[j][0] for j in range(len(valor_wc))])
            spear_lr.append([metrlr[k][1] for k in range(len(valor_wc))])
            pear_err_lr.append([metrlr[j][2] for j in range(len(valor_wc))])
            kend_lr.append([metrlr[k][3] for k in range(len(valor_wc))])
            tqdm.write(f"iteration_global={i}")
        arrays_lr=[np.array(rat_err_lr),np.array(spear_lr),np.array(pear_err_lr),np.array(kend_lr)]
        colum=mean_columns_list(np.array(matrix_scores_last_rounds_re))
        compar_list=[true_sv]+colum
        return arrays_lr,compar_list,metrlr
    
    def simulations_retraining(self,iteraci=10,iter_fed=10,epochs=1):
            if self.partition_type=="N-IID":
                path1=f'./RESULTS/{self.dataset_name}/{self.num_clients}cli({self.alpha})_{iter_fed}fed_retrain'
                title=f"Case N-IID, alpha={self.alpha}, and iter={iter_fed}"
                os.makedirs(path1, exist_ok=True)
            elif self.partition_type=='IID':
                path1=f'./RESULTS/{self.dataset_name}/{self.num_clients}cli({self.partition_type})_{iter_fed}fed_retrain'
                title=f"Case {self.partition_type}, and iter={iter_fed}"
                os.makedirs(path1, exist_ok=True)
            name0=f"values_clients"
            name2=f"correlations"
            metricas_global=["Normalize_lse","Spearman","Pearson","Kendall"]
            transform_re=self.statistic_to_retr_game(path1,iteraci,iter_fed,epochs=epochs)
            last_round_global=transform_re[1]
            combine_plot(last_round_global,["retr_SV","I1I","L1O","IE2I","LE2O","SE","EE","PPCE","f_I1I","f_L1O","Data_SV"],path1,title,name0)
            last_round_metr_global=transform_re[2]
            plot_coeficits(last_round_metr_global,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE","f_I1I","f_L1O","Data_SV"],path1,title,name2)
            self.boxplot(transform_re[0],metricas_global,path1,"Comparation_to_retraining_SV",["I1I","L1O","IE2I","LE2O","SE","EE","PPCE","f_I1I","f_L1O"],retraining="retraining")
            return transform_re[0]


def main(dict):
    for i in dict.keys():
        mol, data_name, num_cli, par_type,alpha,iter_global,iter_fed, epochs,retrining=dict[i]
        juegos=games(mol,data_name,num_cli,par_type,alpha) 
        if retrining==None:
            play=juegos.simulation_global_iter(iter_global,iter_fed,epochs)
            valores=play[0]
            if par_type=="N-IID":
                path1=f'./RESULTS_local/{data_name}/{num_cli}cli({alpha})_{iter_fed}fed/values/scores_over_glo_iter.log'
                logger_f(f"{data_name}, num_cli={num_cli}, alpha={alpha}, iter_global={iter_global}, iter_fed={iter_fed}, {play[1]}",path1)
                names=["Nor_lse","Spearman","Pearson","Kendall"]
                for i in range(4):
                    path=f'./RESULTS_local/{data_name}/{num_cli}cli({alpha})_{iter_fed}fed/values/{names[i]}.log'
                    logger_f(f"{data_name}, num_cli={num_cli}, alpha={alpha}, iter_global={iter_global}, iter_fed={iter_fed}, {valores[0][i]}",path)
            elif par_type=="IID":
                path1=f'./RESULTS_local/{data_name}/{num_cli}cli({par_type})_{iter_fed}fed/values/scores_over_glo_iter.log'
                logger_f(f"{data_name}, num_cli={num_cli}, {par_type}, iter_global={iter_global}, iter_fed={iter_fed}, {play[1]}",path1)
                names=["Nor_lse","Spearman","Pearson","Kendall"]
                for i in range(4):
                    path=f'./RESULTS_local/{data_name}/{num_cli}cli({par_type})_{iter_fed}fed/values/{names[i]}.log'
                    logger_f(f"{data_name}, num_cli={num_cli}, {par_type}, iter_global={iter_global}, iter_fed={iter_fed}, {valores[0][i]}",path)
        elif retrining=="ON":
            juegos.simulations_retraining(iter_global,iter_fed,epochs)
        else:
            juegos.simulation_cosine_sv(iter_global,iter_fed,epochs)



if __name__ == "__main__":
    # juegos=games("LOG_S", "STROKE",6,"N-IID",0.5)
    # path="./RESULTS_local"
    # dicts=juegos.local_coalicional_value_at_round(10,path,8)
    # juegos.coalition_plot(dicts[0],dicts[1])
    # values_local=juegos.local_pric_ce(dicts[0],path)
    # # values_global=pri_ce(dicts[1],path)
    # # combine_plot(values_global,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path,"coef_global","coef_global")
    # combine_plot(values_local,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path,"coef_local","coef_local")





    # #format:[model, data_name, num_client, data_partition, alpha, global_iter, fed_iter, local_epochs, None]
    # #if you wanna see results comparing to the retraining game change None to "retraining"
    parser = argparse.ArgumentParser(description="Parse command-line arguments example.")

    # Add arguments
    parser.add_argument("--num", type=int, default=0, help="Exp num")

    # Parse the arguments
    args = parser.parse_args()

    #dict_exper={
     #   0:["MLP","mnist",3,"IID",0.5,1,1,1,None], #This is for alpha = 0.1
        #0:["CNN_brain","BRAIN",6,"IID",0.5,10,5,5,None], # This is for 5 fed. iterations
        # 1:["CNN_brain","BRAIN",6,"N-IID",0.1,10,10,5,None], #This is for alpha = 0.1
        #2:["CNN_brain","BRAIN",6,"IID",0.5,10,15,5,None], # This is for 15 fed. iterations
        #3:["CNN_brain","BRAIN",6,"N-IID",1.0,10,10,5,None], #This is for alpha = 1.0
        #4:["CNN_brain","BRAIN",3,"IID",0.5,10,10,5,None], #This is for 3 clients
        #5:["CNN_brain","BRAIN",9,"IID",0.5,10,10,5,None], #This is for 9 clients
    #}


    dict_exper={
                 0:["CNN","cifar10",9,"N-IID",0.5,3,5,5,None],
                 #1:["CNN_brain","BRAIN",9,"N-IID",0.5,10,10,5,None],
               }
    
    tmp = {args.num: dict_exper[args.num]}
    main(tmp)
    #main(dict_exper) 


    