import random
import torch
import numpy as np
import torch.nn as nn
import math
from tqdm import tqdm 
from itertools import combinations
from client_fed_local import federation
from utils import transform_dict_to_lists, shapley,pri_ce,combine_plot,plot_coeficits, cosine_similarity_models, effect
from utils import approx_quantities
from loggers import logger_f#metricslogger
from utils import plot_boxplot_with_stats
import os


torch.cuda.is_available()
# dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)

#This class contain the functions to be run. The main function to be consider is the one call statistic, with the 
#output of this function with can compute the mean and std of each ppce with respect to the sv and the values of 
#the ppce and sv of each client at the last iteration of fed and the sv anf ppce accumulative over rounds


class games(federation):
    def __init__(self, model_type, dataset_name, num_clients,partition_type,alpha,device=dev):
        super().__init__(model_type, dataset_name, num_clients,partition_type,alpha,device=dev)

    
    def coalitional_values_II(self,iteration, logger_file="results",epochs=5):
        """Compute the evaluation of each coalition at iteration"""
        for client in range(self.num_clients):
            self.lista_clientes[client].reset_parameters()
        self.reset_parameters
        d = dict()
        d[()]=0
        for i in range(iteration):
            self.traing_together(epochs)
        for client in range(self.num_clients):
            d[tuple([client])] = self.lista_clientes[client].evaluation(self.test_global)
        for subset_len in range(2,self.num_clients+1):
            for subset_idx in combinations(list(range(self.num_clients)), subset_len):
                part=[self.lista_clientes[i] for i in subset_idx]
                params = self.weighted_aggre(part)
                self.set_parameters(params)
                d[subset_idx] = self.evaluation_global(self.test_global)
            tqdm.write(f"subset size {subset_len}")
        logger_f(f"at coalitional values: {d}",logger_file)
        trans=transform_dict_to_lists(d,self.num_clients)
        return trans
    
    
    
    def traing_together(self, num_epochs):
        """sort of function that does  fedAvg"""
        for client in self.lista_clientes:
            weights_modelo = self.model.state_dict()
            client.load_state_dict(weights_modelo)
            client.fit(num_epochs)
        params = self.weighted_aggre(self.lista_clientes)
        self.set_parameters(params)
       

    
    def coalicional_value_at_round(self,iter,logger_file="results",epochs=5):
        """
        iter:iteracion
        output: dictionary. keys are tuples that represent each coalition, values are the evaluation of the model of each coalition.  
        """
        self.traing_together(epochs)
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
        tqdm.write(f"iteration_fed={iter}")
        logger_f(f"at coalitional values: {d}",logger_file)
        return d

    def cosine_sim_clie_server(self):
        co_si=[]
        for client in self.lista_clientes:
            value=cosine_similarity_models(client.model,self.model)
            co_si.append(value)
        logger_f(f"Cosine: {co_si}","cosine")
        return co_si

    

    def valores_por_rondas(self,itera,base,path,epochs=5):
            """
            itera: number of federation iterations 
            output: two lists. The first one contains the sv and ppce for each client at the last round. 
            The second list contains the sv and ppce cumulative over the all itera (cummulative means the mean)
            """
            for client in range(self.num_clients):
              self.lista_clientes[client].reset_parameters()
            self.reset_parameters()
            avg_base=[]
            avg_i1i=[]
            avg_l1o=[]
            avg_ieei=[]
            avg_leeo=[]
            avg_se=[]
            avg_ee=[]
            avg_ppce=[]
            for i in range(itera):
                dic_values=self.coalicional_value_at_round(i,epochs=epochs)
                trans=transform_dict_to_lists(dic_values,self.num_clients)
                priv_pr=pri_ce(self.num_clients,trans[0],trans[1],path)
                if base=="shapley":
                    base_metr=shapley(self.num_clients,trans[0],trans[1],path)
                elif base=="cosine":
                    base_metr=self.cosine_sim_clie_server()
                avg_base.append(base_metr)
                avg_i1i.append(priv_pr[0])
                avg_l1o.append(priv_pr[1])
                avg_ieei.append(priv_pr[2])
                avg_leeo.append(priv_pr[3])
                avg_se.append(priv_pr[4])
                avg_ee.append(priv_pr[5])
                avg_ppce.append(priv_pr[6])
                if i==itera-1:
                    efecto=effect(self.num_clients,trans[0],trans[1])#(it should be a mean an stardard deviation)
            games_pp=[avg_base[itera-1],avg_i1i[itera-1],avg_l1o[itera-1],avg_ieei[itera-1],avg_leeo[itera-1],avg_se[itera-1],avg_ee[itera-1],avg_ppce[itera-1]]
            mean_sv=np.mean(avg_base,axis=0).tolist()
            mean_i1i=np.mean(avg_i1i,axis=0).tolist()
            mean_l1o=np.mean(avg_l1o,axis=0).tolist()
            mean_ieei=np.mean(avg_ieei,axis=0).tolist()
            mean_leeo=np.mean(avg_leeo,axis=0).tolist()
            mean_se=np.mean(avg_se,axis=0).tolist()
            mean_ee=np.mean(avg_ee,axis=0).tolist()
            mean_ppce=np.mean(avg_ppce,axis=0).tolist()
            mean_values=[mean_sv,mean_i1i,mean_l1o, mean_ieei,mean_leeo,mean_se,mean_ee,mean_ppce]
            return games_pp,mean_values,efecto

    def statistic(self,path,iter_one,iter_two=10,base="shapley",epochs=5):
        """
        iter_one:number of iteration to run the federation several times 
        iter_two:number of federation iterations
        output1: matrix with each row the  
        """
        rat_err_lr=[]
        spear_lr=[]
        rat_err_mean=[]
        spear_mean=[]
        pear_err_lr=[]
        kend_lr=[]
        pear_err_mean=[]
        kend_mean=[]
        efIE2I=np.zeros((self.num_clients,self.num_clients))
        efLE2O=np.zeros((self.num_clients,self.num_clients))
        efEE=np.zeros((self.num_clients,self.num_clients))
        for i in range(iter_one):
            valores=self.valores_por_rondas(iter_two,base,path,epochs)
            efIE2I+=valores[2][0]
            efLE2O+=valores[2][1]
            efEE+=valores[2][2]
            #logger_results
            logger_f(f"{self.dataset_name}, num_cli: {self.num_clients}, alpha: {self.alpha}, iteration: {i}","results")
            #logger.info(f"{self.dataset_name}, num_cli: {self.num_clients}, alpha: {self.alpha}, iteration: {i}")
            #logger_shapley
            logger_f(f"{self.dataset_name}, num_cli: {self.num_clients}, alpha: {self.alpha}, iteration: {i}",base)
            #svlogger.info(f"{self.dataset_name}, num_cli: {self.num_clients}, alpha: {self.alpha}, iteration: {i}")
            #logger_ppce
            logger_f(f"{self.dataset_name}, num_cli: {self.num_clients}, alpha: {self.alpha}, iteration: {i}","ppce")
            #ppcelogger.info(f"{self.dataset_name}, num_cli: {self.num_clients}, alpha: {self.alpha}, iteration: {i}")
            metrlr=approx_quantities(valores[0][0],valores[0][1:],f"{path}/values/metrics_global.log")
            metrmean=approx_quantities(valores[1][0],valores[1][1:],f"{path}/values/metrics_global.log")
            #logger_metrics
            logger_f(f"{self.dataset_name}, num_cli: {self.num_clients}, alpha: {self.alpha}, iteration: {i}","metrics")
            #metricslogger.info(f"{self.dataset_name}, num_cli: {self.num_clients}, alpha: {self.alpha}, iteration: {i}")
            rat_err_lr.append([metrlr[j][0] for j in range(7)])
            spear_lr.append([metrlr[k][1] for k in range(7)])
            rat_err_mean.append([metrmean[j][0] for j in range(7)])
            spear_mean.append([metrmean[k][1] for k in range(7)])
            pear_err_lr.append([metrlr[j][2] for j in range(7)])
            kend_lr.append([metrlr[k][3] for k in range(7)])
            pear_err_mean.append([metrmean[j][2] for j in range(7)])
            kend_mean.append([metrmean[k][3] for k in range(7)])
            tqdm.write(f"iteration_global={i}")
        effects_EE=[efIE2I/iter_one,efLE2O/iter_one,efEE/iter_one]
        arrays_lr=[np.array(rat_err_lr),np.array(spear_lr),np.array(pear_err_lr),np.array(kend_lr)]
        arrays_mean=[np.array(rat_err_mean),np.array(spear_mean),np.array(pear_err_mean),np.array(kend_mean)]
        return arrays_lr,arrays_mean,valores,metrlr,metrmean,effects_EE

        

    def boxplot(self,valores,path,name,metricas=["Normalize_lse","Spearman","Pearson","Kendall"]):
        # valores=self.valores_por_rondas(ite)
        for i in range(len(valores)):
            metric=metricas[i]
            avg_i1i=[row[0] for row in valores[i]]
            avg_l1o=[row[1] for row in valores[i]]
            avg_ieei=[row[2] for row in valores[i]]
            avg_leeo=[row[3] for row in valores[i]]
            avg_se=[row[4] for row in valores[i]]
            avg_ee=[row[5] for row in valores[i]]
            avg_ppce=[row[6] for row in valores[i]]
            big_list=[avg_i1i,avg_l1o,avg_ieei,avg_leeo,avg_se,avg_ee,avg_ppce]
            means = [np.mean(lst).item() for lst in big_list]
            stds = [np.std(lst).item() for lst in big_list]
            logger_f(f"means for {metric}: {means}","mean_std")
            logger_f(f"stds for {metric}: {stds}","mean_std")
            #mean_stdlogger.info(f"means for {metric}: {means}")
            #mean_stdlogger.info(f"stds for {metric}: {stds}")
            plot_boxplot_with_stats(big_list,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path,f"{metric}: {name}",f"{metric}: {name}")
        logger_f(f"{name}, {self.dataset_name}, {self.num_clients}, {self.alpha}","mean_std")


        

    def valor_rondas(self,itera,epochs=5):
            """
            itera: number of federation iterations 
            output: two lists. The first one contains the sv and ppce for each client at the last round. 
            The second list contains the sv and ppce cumulative over the all itera (cummulative means the mean)
            """
            #for client in range(self.num_clients):
              #self.lista_clientes[client].reset_parameters()
            #self.reset_parameters()
            avg_sv=[]
            avg_i1i=[]
            avg_l1o=[]
            avg_ieei=[]
            avg_leeo=[]
            avg_se=[]
            avg_ee=[]
            avg_ppce=[]
            rat_err=[]
            spear=[]
            for i in range(itera):
                dic_values=self.coalicional_value_at_round(i,epochs)
                trans=transform_dict_to_lists(dic_values,self.num_clients)
                priv_pr=pri_ce(self.num_clients,trans[0],trans[1])
                shapley_ro=shapley(self.num_clients,trans[0],trans[1])
                metr=approx_quantities(shapley_ro,priv_pr)
                rat_err.append([metr[j][0] for j in range(7)])
                spear.append([metr[k][1] for k in range(7)])
                avg_sv.append(shapley_ro)
                avg_i1i.append(priv_pr[0])
                avg_l1o.append(priv_pr[1])
                avg_ieei.append(priv_pr[2])
                avg_leeo.append(priv_pr[3])
                avg_se.append(priv_pr[4])
                avg_ee.append(priv_pr[5])
                avg_ppce.append(priv_pr[6])
            games_pp=[avg_sv[itera-1],avg_i1i[itera-1],avg_l1o[itera-1],avg_ieei[itera-1],avg_leeo[itera-1],avg_se[itera-1],avg_ee[itera-1],avg_ppce[itera-1]]
            mean_sv=np.mean(avg_sv,axis=0).tolist()
            mean_i1i=np.mean(avg_i1i,axis=0).tolist()
            mean_l1o=np.mean(avg_l1o,axis=0).tolist()
            mean_ieei=np.mean(avg_ieei,axis=0).tolist()
            mean_leeo=np.mean(avg_leeo,axis=0).tolist()
            mean_se=np.mean(avg_se,axis=0).tolist()
            mean_ee=np.mean(avg_ee,axis=0).tolist()
            mean_ppce=np.mean(avg_ppce,axis=0).tolist()
            mean_values=[mean_sv,mean_i1i,mean_l1o, mean_ieei,mean_leeo,mean_se,mean_ee,mean_ppce]
            array=[np.array(rat_err),np.array(spear)]
            return games_pp,mean_values,array
    
    def simulation_global_iter(self,iteraci=10,iter_fed=10,base="shapley",epochs=5):
        if self.partition_type=="N-IID":
            path1=f'./PLOTS/{self.dataset_name}/{self.num_clients}cli({self.alpha})_{base}_{iter_fed}fed'
            title=f"Case N-IID, alpha={self.alpha}, and iter={iter_fed}"
            os.makedirs(path1, exist_ok=True)
        elif self.partition_type=='IID':
            path1=f'./PLOTS/{self.dataset_name}/{self.num_clients}cli({self.partition_type})_{base}_{iter_fed}fed'
            title=f"Case {self.partition_type}, and iter={iter_fed}"
            os.makedirs(path1, exist_ok=True)
        name1=f"values_clients"
        name2=f"correlations"
        name3=f"values_clients_mean"
        name4=f"correlations_mean"
        transform_re=self.statistic(path1,iteraci,iter_fed,base,epochs=epochs)
        if self.partition_type=="N-IID":
            logger_f(f"{self.dataset_name}, num_cli: {self.num_clients}, alpha: {self.alpha},fed_ite: {iter_fed}, effect: {transform_re[5]}","mean_effect")
        else:
            logger_f(f"{self.dataset_name}, num_cli: {self.num_clients}, IID, fed_ite: {iter_fed}, effect: {transform_re[5]}","mean_effect")
        last_round=transform_re[2][0]
        mean_over=transform_re[2][1]
        combine_plot(last_round,[f"{base}","I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name1)
        last_round_metr=transform_re[3]#approx_quantities(last_round[0],last_round[1:])
        plot_coeficits(last_round_metr,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name2)
        mean_rounds=transform_re[4]#approx_quantities(mean_over[0],mean_over[1:])
        combine_plot(mean_over,[f"{base}","I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name3)
        plot_coeficits(mean_rounds,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name4)
        self.boxplot(transform_re[0],path1,"last_round")
        self.boxplot(transform_re[1],path1,"mean_over_rounds")

    def simulation_fed_iter(self,iter_fed=10):
        path1=f'./PLOTS/{self.dataset_name}/{self.num_clients}clients({self.alpha})stad_round'
        title=f"Diric.part. alpha={self.alpha}, and iter={iter_fed}"
        name1=f"values_clients"
        name2=f"correlations"
        name3=f"values_clients_mean"
        name4=f"correlations_mean"
        transform_re=self.valor_rondas(iter_fed,5)
        last_round=transform_re[0]
        mean_over=transform_re[1]
        combine_plot(last_round,["SV","I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name1)
        last_round_metr=approx_quantities(last_round[0],last_round[1:])
        plot_coeficits(last_round_metr,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name2)
        mean_rounds=approx_quantities(mean_over[0],mean_over[1:])
        combine_plot(mean_over,["SV","I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name3)
        plot_coeficits(mean_rounds,["I1I","L1O","IE2I","LE2O","SE","EE","PPCE"],path1,title,name4)
        self.boxplot(transform_re[2],path1,"over_fed_rounds")



    
if __name__ == "__main__":
    #class param
    mol="CNN_brain"
    data_name="BRAIN"
    par_type="N-IID"
    num_cli=6
    alpha=0.5
    #func param
    iter_global=10
    iter_fed=10
    base="shapley"
    epochs=5
    ###
    juegos=games(mol,data_name,num_cli,par_type,alpha) 
    juegos.simulation_global_iter(iter_global,iter_fed,base,epochs)
    # mol="CNN_brain"
    # data_name="BRAIN"
    # par_type="N-IID"
    # num_cli=6
    # alpha=0.5
    # #func param
    # iter_global=10
    # iter_fed=10
    # base="cosine"
    # epochs=1
    # ###
    # juegos=games(mol,data_name,num_cli,par_type,alpha) 
    # juegos.simulation_global_iter(iter_global,iter_fed,base,epochs)
    # setting1=[5,6]#fed_ite, num_cli
    # setting2=[10,9]
    # setting=[setting1,setting2]
    # for sett in setting:
    # mol="LOGISTIC"
    # data_name="BREAST"
    # par_type="N-IID"
    # num_cli=3
    # alpha=0.5
    # #func param
    # iter_global=1
    # iter_fed=1
    # base="cosine"
    # epochs=1
    # ###
    # juegos=games(mol,data_name,num_cli,par_type,alpha) 
    # juegos.simulation_global_iter(iter_global,iter_fed,base,epochs)
