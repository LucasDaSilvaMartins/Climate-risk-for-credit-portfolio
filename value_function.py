import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import integrate
import warnings
from scipy.integrate import IntegrationWarning
from scipy.stats import gaussian_kde


warnings.filterwarnings("ignore", category=IntegrationWarning)
warnings.simplefilter("ignore")


class Firm_1: #None proportional jumps
    def __init__(self,prod_0,r,sig,a,b,c,w_1,w_2,T,alpha,eta,scenario,sector,AP,beta,lambda_0,region,time,scenario_temp):
        """
        Initialize the set of parameters for the portfolio loss.

        Parameters:
        -------------------------------------------------------------------------------------------------------------------------------------
        prod_0 : float
            Initial log production of the firm, must be positive
        r : float
            Interest rate, must bu positive.
        sig : float
            Volatility modeling uncertaity in production, must be positive.
        a : float
            Average production level (without emission). Constant term in the drift function.
        b : float
            Mean-reverting parameter
        c : float
            Average production increase by increasing emission
        w_1 : float
            Reward coefficient, must be positive and stricly lower than eta
        w_2 : float
            Penalty coefficient
        T : float
            Time maturity/horizon.
        alpha : float/array (if it is a function)
            Linear coefficient for cost function 
        eta : float/array (if it is a function)
            Quadratic coefficient for cost function 
        scenario: list of str
            Climate scenarios (SSP) name.
        sector: str
            Sector name.
        AP : float
            Average price of sell
        beta : matrix
            Jump size distribution for different time t
        lambda_0 : float
            Initial intensity of jumps
        region : str
            Region of the firm
        time : np.ndarray
            Time grid


        """
        self.prod_0=prod_0
        self.r=r
        self.sig=sig
        self.a=a
        self.b=b
        self.c=c
        self.w_1=w_1
        self.w_2=w_2
        self.T=T
        self.alpha=alpha
        self.eta=eta
        self.sector=sector
        self.scenario=scenario
        self.AP=AP
        self.beta=beta
        self.lambda_0=lambda_0
        self.region=region
        self.time=time
        self.scenario_temp=scenario_temp
        self.n=len(self.time)

        self.SCENARIOS=["SSP1-26",
                "SSP2-45",
                "SSP3-70 (Baseline)",
                "SSP4-60",
                "SSP5-85 (Baseline)"]

        self.SCENARIOS_temp=["SSP1 - 2.6","SSP2 - 4.5",
        "SSP3 - Baseline","SSP4 - 6.0","SSP5 - Baseline"]
        #Load SSP data
        climate_data_path="/Users/lucas/Documents/Doctorat/Production/Climate risk for credit portfolio/data/SSP_CMIP6_201811.csv"
        df_ssp=pd.read_csv(climate_data_path,sep=',')

        #TEMPERATURE:

        #Load Temperature data
        energies_data_path='/Users/lucas/Documents/Doctorat/Production/Climate risk for credit portfolio/data/owid-ipcc-scenarios.csv'
        df_energies=pd.read_csv(energies_data_path,sep=';')
        df_temp=df_energies[["Scenario","Temperature","Year"]]
        df_temp=df_temp[df_temp["Scenario"].isin(self.SCENARIOS_temp)]

        #Bulding a dictionnary for each scenario
        self.dict_temp={}
        for scen in self.SCENARIOS_temp:
            self.dict_temp[f"df_temp_{scen}"]=df_temp[df_temp["Scenario"]==scen]
        years_temp=self.dict_temp[f"df_temp_{scen}"]["Year"]
        self.years_temp_int=[int(y) for y in years_temp]
        self.years_temp_int_smooth=np.linspace(np.min(self.years_temp_int),np.max(self.years_temp_int),self.n)

        #Interpolate
        self.dict_temp_interpolate={}
        for scen in self.SCENARIOS_temp:
            cs=CubicSpline(self.years_temp_int,self.dict_temp[f"df_temp_{scen}"]["Temperature"])
            self.dict_temp_interpolate[f"df_temp_interpolate_{scen}"]=cs(self.years_temp_int_smooth)
            
        # EMISSION:

        #Years
        years_label=["2015","2020","2030","2040","2050","2060","2070","2080","2090","2100"]
        years=[int(y) for y in years_label]
        ssp_time=np.sort(np.array(years-np.min(years)))

        #Filter for a given sector and region
        df0=df_ssp[(df_ssp["REGION"]==region)&(df_ssp["VARIABLE"]==f"CMIP6 Emissions|CO2|{sector}")].reset_index(drop=True)
        df=df0[df0["SCENARIO"].isin(self.SCENARIOS)].reset_index(drop=True)
        self.ssp_value={}
        self.ssp_value_interpolate={}
        B_t=(self.AP/(self.r+self.b)*(1-np.exp(-(self.b+self.r)*(self.T)))*self.c-self.alpha[0])/(2*self.eta[0]) # B(0)
        self.years_smooth = np.linspace(np.min(years), np.max(years), self.n)

        for scen in self.SCENARIOS:
            self.ssp_value[f"ssp_value_{scen}"]=df.loc[df["SCENARIO"] == scen, years_label].iloc[0]
            self.ssp_value[f"ssp_value_{scen}"]=self.ssp_value[f"ssp_value_{scen}"].values.astype(float)
            self.ssp_value[f"ssp_value_{scen}"]=B_t*self.ssp_value[f"ssp_value_{scen}"]/self.ssp_value[f"ssp_value_{scen}"][0]

            #Interpolation scenario
            cs=CubicSpline(years,self.ssp_value[f"ssp_value_{scen}"])
            self.ssp_value_interpolate[f"ssp_value_interpolate_{scen}"]=cs(self.years_smooth)

    def input_emission(self):
        return self.ssp_value_interpolate[f"ssp_value_interpolate_{self.scenario}"]

    def B_t(self):
        return self.AP/(self.r+self.b)*(1-np.exp(-(self.b+self.r)*(self.T-self.time)))

    def gamma_optimal_unconstrained(self,B_t=None):
        if B_t is None:
            B_t=self.B_t()
        return 1/(2*self.eta)*(B_t*self.c-self.alpha)

    def gamma_optimal_control(self,s,B_t=None):
        if B_t is None:
            B_t=self.B_t()
        Theta_t=1/(2*self.eta)*(B_t*self.c-self.alpha)
        gam_excess=np.maximum(((self.ssp_value_interpolate[f"ssp_value_interpolate_{s}"]-Theta_t)),0)
        gam_lack=np.maximum((Theta_t-self.ssp_value_interpolate[f"ssp_value_interpolate_{s}"]),0)
        return np.maximum(1/(2*self.eta)*(B_t*self.c-self.alpha-2*self.w_2/(1+self.w_2/self.eta)*gam_excess-2*self.w_1/(1-self.w_1/self.eta)*gam_lack),0)

    def _Gamma_function(self,s,B_t=None):
        if B_t is None:
            B_t=self.B_t()
        return self.w_1*np.maximum(self.ssp_value_interpolate[f"ssp_value_interpolate_{s}"]-self.gamma_optimal_control(s,B_t),0)-self.w_2*np.maximum(self.gamma_optimal_control(s,B_t)-self.ssp_value_interpolate[f"ssp_value_interpolate_{s}"],0)+B_t*self.c*self.gamma_optimal_control(s,B_t)-self.alpha*self.gamma_optimal_control(s,B_t)-self.eta*self.gamma_optimal_control(s,B_t)**2

    def DICE_function(self,u):
        return 0.0028388*u**2

    def lambda_t(self,t,scenario):
        k=min(int(t*self.n/85),self.n-1)
        u=self.dict_temp_interpolate[f"df_temp_interpolate_{scenario}"][k]
        t_ref=self.dict_temp_interpolate[f"df_temp_interpolate_{scenario}"][0]
        return(self.lambda_0*self.DICE_function(u)/self.DICE_function(t_ref))

    def trajectoire_lambda(self,scenario):
        lam=np.empty(self.n)
        for k,t in enumerate(self.time):
            lam[k]=self.lambda_t(t,scenario)
        return lam
    
    def _integrande(self,t_0):
        Jump=1
        index=self.time>=t_0
        time=self.time[index]
        lambda_traj=self.trajectoire_lambda(self.scenario_temp)
        return(np.exp(-self.r*(time-t_0))*(self.B_t()[index]*self.a-lambda_traj[index]*Jump+self._Gamma_function(self.scenario)[index])) 

    def value_funcion(self,p,t):
        k=min(int(t*self.n/85),self.n-1)
        inte=np.sum(self._integrande(t)[t:]*self.time[1])
        return inte+p*self.B_t()[k]
    
    def _trajectoire_pt(self,scenario,scenario_temp,gamma_optimal_control=None):
        if gamma_optimal_control is None:
            gamma_optimal_control=self.gamma_optimal_control(scenario)
        p=np.empty(self.n)
        p[0]=self.prod_0
        Jump=1 # Jump size
        dt=self.time[1]
        for i in range(1,self.n):
            p[i]=p[i-1]+(self.a-self.b*p[i-1]+self.c*gamma_optimal_control[i-1])*dt+self.sig*np.sqrt(dt)*np.random.randn()-Jump*np.random.poisson(self.lambda_t(self.time[i-1],scenario_temp)*dt) #modifier ici si on change les sauts (le j'ai saut de 1)
        return p

    def trajectoire_value_function(self,scenario,scenario_temp,_trajectoire_pt=None,gamma_optimal_control=None):
        if gamma_optimal_control is None:
            gamma_optimal_control=self.gamma_optimal_control(scenario)
        if _trajectoire_pt is None:
            _trajectoire_pt=self._trajectoire_pt(scenario,scenario_temp,gamma_optimal_control)
        trajectoire=np.empty(self.n)
        _trajectoire_pt=self._trajectoire_pt(scenario,scenario_temp)
        for i in range (self.n):
            trajectoire[i]=self.value_funcion(_trajectoire_pt[i],int(85/self.n*i))
        return trajectoire

    def multi_value(self,scenario,scenario_temp,n,t):
        k=min(int(t*self.n/85),self.n-1)
        value_matrix=np.empty(n)
        gamma_optimal_control=self.gamma_optimal_control(scenario)
        for j in range (n):
            _trajectoire_pt=self._trajectoire_pt(scenario,scenario_temp,gamma_optimal_control)[k]
            value_matrix[j]=_trajectoire_pt
        value_matrix=value_matrix+self.B_t()[k]
        return value_matrix

    def threshold(self,scenario,scenario_temp,PD,t,n=10000):# Ajouter interval de confiance et le calculer sans risque de transition et sans saut
        distrib=self.multi_value(scenario,scenario_temp,t,n)
        return np.quantile(distrib,PD)
    

class Firm_2: #Proportional jumps
    def __init__(self,prod_0,r,sig,a,b,c,w_1,w_2,T,alpha,eta,scenario,sector,AP,beta,lambda_0,region,time,scenario_temp):
        """
        Initialize the set of parameters for the portfolio loss.

        Parameters:
        -------------------------------------------------------------------------------------------------------------------------------------
        prod_0 : float
            Initial log production of the firm, must be positive
        r : float
            Interest rate, must bu positive.
        sig : float
            Volatility modeling uncertaity in production, must be positive.
        a : float
            Average production level (without emission). Constant term in the drift function.
        b : float
            Mean-reverting parameter
        c : float
            Average production increase by increasing emission
        w_1 : float
            Reward coefficient, must be positive and stricly lower than eta
        w_2 : float
            Penalty coefficient
        T : float
            Time maturity/horizon.
        alpha : float/array (if it is a function)
            Linear coefficient for cost function 
        eta : float/array (if it is a function)
            Quadratic coefficient for cost function 
        scenario: list of str
            Climate scenarios (SSP) name.
        sector: str
            Sector name.
        AP : float
            Average price of sell
        beta : matrix
            Jump size distribution for different time t
        lambda_0 : float
            Initial intensity of jumps
        region : str
            Region of the firm
        time : np.ndarray
            Time grid

        """
        self.prod_0=prod_0
        self.r=r
        self.sig=sig
        self.a=a
        self.b=b
        self.c=c
        self.w_1=w_1
        self.w_2=w_2
        self.T=T
        self.alpha=alpha
        self.eta=eta
        self.sector=sector
        self.scenario=scenario
        self.AP=AP
        self.beta=beta
        self.lambda_0=lambda_0
        self.region=region
        self.time=time
        self.scenario_temp=scenario_temp
        self.n=len(self.time)


        self.SCENARIOS=["SSP1-26",
                "SSP2-45",
                "SSP3-70 (Baseline)",
                "SSP4-60",
                "SSP5-85 (Baseline)"]

        self.SCENARIOS_temp=["SSP1 - 2.6","SSP2 - 4.5",
        "SSP3 - Baseline","SSP4 - 6.0","SSP5 - Baseline"]
        #Load SSP data
        climate_data_path="/Users/lucas/Documents/Doctorat/Production/Climate risk for credit portfolio/data/SSP_CMIP6_201811.csv"
        df_ssp=pd.read_csv(climate_data_path,sep=',')

        #TEMPERATURE:

        #Load Temperature data
        energies_data_path='/Users/lucas/Documents/Doctorat/Production/Climate risk for credit portfolio/data/owid-ipcc-scenarios.csv'
        df_energies=pd.read_csv(energies_data_path,sep=';')
        df_temp=df_energies[["Scenario","Temperature","Year"]]
        df_temp=df_temp[df_temp["Scenario"].isin(self.SCENARIOS_temp)]

        #Bulding a dictionnary for each scenario
        self.dict_temp={}
        for scen in self.SCENARIOS_temp:
            self.dict_temp[f"df_temp_{scen}"]=df_temp[df_temp["Scenario"]==scen]
        years_temp=self.dict_temp[f"df_temp_{scen}"]["Year"]
        self.years_temp_int=[int(y) for y in years_temp]
        self.years_temp_int_smooth=np.linspace(np.min(self.years_temp_int),np.max(self.years_temp_int),self.n)

        #Interpolate
        self.dict_temp_interpolate={}
        for scen in self.SCENARIOS_temp:
            cs=CubicSpline(self.years_temp_int,self.dict_temp[f"df_temp_{scen}"]["Temperature"])
            self.dict_temp_interpolate[f"df_temp_interpolate_{scen}"]=cs(self.years_temp_int_smooth)
            
        # EMISSION:

        #Years
        years_label=["2015","2020","2030","2040","2050","2060","2070","2080","2090","2100"]
        years=[int(y) for y in years_label]
        ssp_time=np.sort(np.array(years-np.min(years)))

        #Filter for a given sector and region
        df0=df_ssp[(df_ssp["REGION"]==region)&(df_ssp["VARIABLE"]==f"CMIP6 Emissions|CO2|{sector}")].reset_index(drop=True)
        df=df0[df0["SCENARIO"].isin(self.SCENARIOS)].reset_index(drop=True)
        self.ssp_value={}
        self.ssp_value_interpolate={}
        B_t=(self.AP/(self.r+self.b)*(1-np.exp(-(self.b+self.r)*(self.T)))*self.c-self.alpha[0])/(2*self.eta[0]) 
        self.years_smooth = np.linspace(np.min(years), np.max(years), self.n)

        for scen in self.SCENARIOS:
            self.ssp_value[f"ssp_value_{scen}"]=df.loc[df["SCENARIO"] == scen, years_label].iloc[0]
            self.ssp_value[f"ssp_value_{scen}"]=self.ssp_value[f"ssp_value_{scen}"].values.astype(float)
            self.ssp_value[f"ssp_value_{scen}"]=B_t*self.ssp_value[f"ssp_value_{scen}"]/self.ssp_value[f"ssp_value_{scen}"][0]

            #Interpolation scenario
            cs=CubicSpline(years,self.ssp_value[f"ssp_value_{scen}"])
            self.ssp_value_interpolate[f"ssp_value_interpolate_{scen}"]=cs(self.years_smooth)

    def input_emission(self):
        return self.ssp_value_interpolate[f"ssp_value_interpolate_{self.scenario}"]

    def beta_t_xi(self,scenario_temp):
        beta=np.ones(self.n)
        return beta

    def h(self,scenario_temp):
        return self.b+self.r+self.trajectoire_lambda(scenario_temp)*self.beta_t_xi(scenario_temp)

    def H(self, scenario_temp):
        dt = self.time[1] - self.time[0]
        h = self.h(scenario_temp)
        H = np.empty(self.n, dtype=float)
        H[0] = 0.0
        H[1:] = np.cumsum(0.5 * (h[:-1] + h[1:]) * dt)
        return H

    def B_grid(self, scenario_temp):
        dt = self.time[1] - self.time[0]
        H = self.H(scenario_temp)
        exp_minus_H = np.exp(-H)
        tail = np.cumsum(exp_minus_H[::-1]) * dt
        tail = tail[::-1]
        B_grid = self.AP * np.exp(H) * tail
        return B_grid

    def gamma_optimal_unconstrained(self,scenario_temp): #Now, the optimal control unconstrained depend on the jumps
        return (self.B_grid(scenario_temp)*self.c-self.alpha)/(2*self.eta)

    def gamma_optimal_control(self,scenario_temp,s):
        B_t=self.B_grid(scenario_temp)
        Theta_t=1/(2*self.eta)*(B_t*self.c-self.alpha)
        gam_excess=np.maximum((self.ssp_value_interpolate[f"ssp_value_interpolate_{s}"]-Theta_t),0)
        gam_lack=np.maximum((Theta_t-self.ssp_value_interpolate[f"ssp_value_interpolate_{s}"]),0)
        return np.maximum(1/(2*self.eta)*(B_t*self.c-self.alpha-2*self.w_2/(1+self.w_2/self.eta)*gam_excess-2*self.w_1/(1-self.w_1/self.eta)*gam_lack),0)

    def _Gamma_function(self,scenario_temp,s):
        return self.w_1*np.maximum(self.ssp_value_interpolate[f"ssp_value_interpolate_{s}"]-self.gamma_optimal_control(scenario_temp,s),0)-self.w_2*np.maximum(self.gamma_optimal_control(scenario_temp,s)-self.ssp_value_interpolate[f"ssp_value_interpolate_{s}"],0)+self.B_grid(scenario_temp)*self.c*self.gamma_optimal_control(scenario_temp,s)-self.alpha*self.gamma_optimal_control(scenario_temp,s)-self.eta*self.gamma_optimal_control(scenario_temp,s)**2

    def DICE_function(self,u):
        return 0.0028388*u**2

    def lambda_t(self,t,scenario):
        k=min(int(t*self.n/85),self.n-1)
        u=self.dict_temp_interpolate[f"df_temp_interpolate_{scenario}"][k]
        t_ref=self.dict_temp_interpolate[f"df_temp_interpolate_{scenario}"][0]
        return(self.lambda_0*self.DICE_function(u)/self.DICE_function(t_ref))

    def trajectoire_lambda(self,scenario):
        lam=np.empty(self.n)
        for k,t in enumerate(self.time):
            lam[k]=self.lambda_t(t,scenario)
        return lam

    def _integrande(self,scenario,scenario_temp,t_0):
        index=self.time>=t_0
        time=self.time[index]
        return(np.exp(-self.r*(time-t_0))*(self.B_grid(scenario_temp)[index]*self.a+self._Gamma_function(scenario_temp,scenario)[index])) #modifier ici si on change les sauts (le j'ai saut de 1)

    def value_funcion(self,p,t,scenario,scenario_temp,):
        k=min(int(t*self.n/85),self.n-1)
        inte=np.sum(self._integrande(scenario,scenario_temp,t)[t:]*self.time[1])
        return inte+p*self.B_grid(scenario_temp)[k]
    
    def _trajectoire_pt(self,scenario,scenario_temp,gamma_optimal_control=None):
        if gamma_optimal_control is None:
            gamma_optimal_control=self.gamma_optimal_control(scenario_temp,scenario)
        p=np.empty(self.n)
        p[0]=self.prod_0
        dt=self.time[1]
        for i in range(1,self.n):
            p[i]=p[i-1]+(self.a-self.b*p[i-1]+self.c*gamma_optimal_control[i-1])*dt+self.sig*np.sqrt(dt)*np.random.randn()-p[i-1]*self.beta_t_xi(scenario_temp)[i-1]*np.random.poisson(self.lambda_t(self.time[i-1],scenario_temp)*dt) #modifier ici si on change les sauts (le j'ai saut de 1)
        return p

    def trajectoire_value_function(self,scenario,scenario_temp):
        trajectoire=np.empty(self.n)
        traj=self._trajectoire_pt(scenario,scenario_temp)
        for i in range (self.n):
            trajectoire[i]=self.value_funcion(traj[i],int(85/self.n*i),scenario,scenario_temp,)
        return trajectoire

    def multi_value(self, scenario,scenario_temp,n,t):
        k=min(int(t*self.n/85),self.n-1)
        value_matrix=np.empty(n)
        gamma_optimal_control=self.gamma_optimal_control(scenario_temp,scenario)
        for j in range (n):
            _trajectoire_pt=self._trajectoire_pt(scenario,scenario_temp,gamma_optimal_control)[k]
            value_matrix[j]=_trajectoire_pt
        value_matrix=value_matrix+self.B_grid(scenario_temp)[k]
        return value_matrix

    def threshold(self,scenario,scenario_temp,PD,t,n=10000):# Ajouter interval de confiance et le calculer sans risque de transition et sans saut
        distrib=self.multi_value(scenario,scenario_temp,n,t)
        return np.quantile(distrib,PD)
    
"""
Dans le cas 2, le B_t depend des différents scenario maintenant
Pour la partie portefeuille, il faudra prendre en compte le fait qu'il soit possible qu'il y est des jumps climatique individuel et systemique 
Pour gerer ca, il faudrait que trajectoire V_t et trajectoire p_t prennent en entrée des bruits (pour avoir des browniens systemiques et idiosyncratique et pareil pour les sauts)
Mais comment on gere la proximité géographique des différentes entités?
On doit aussi pouvoir gérer les fait que certaine entreprise on des productions susr différentes zone géographiques et donc potentiellement des sauts à plusieurs endroits (comme Gobet)
Comment faire pour ensuite avoir des distribution de de v_t ou de perte du portefeuille vu la lenteur

Il va me falloir plusieurs tracetoire V_t pour une meme entreprise (pour generer des defauts une fois que j'ai le seuil de defaut)
Ensuite il faudra ca pour toute les entreprises du portefeuille pour generer des pertes de portefeuilles cumuler
Si on fait ca pleins de fois on aura la distribution de la pertes du portefeuille (par scenario climatique)
A partir de la on pourra comparer des mesures de risques par scenario climatique au seins du portefeuille
"""



"""
Gauss legendre
Calculer toute les trajectoire une fois et ensuite appeler la fonction
Faire tout en vecteur et pas en calcul sur boucle
"""

class Portfolio:# il faut gérer la corrélation+ dependance risque physique entre les firms, on ne peut pas prendre directement les V_t de la class précédentes
    def __init__(self,EAD,LGD,PD,portfolio,n,t,scenario,scenario_temp):
        self.EAD=EAD
        self.LGD=LGD
        self.portfolio=portfolio
        self.n=n
        self.t=t
        self.scenario=scenario
        self.scenario_temp=scenario_temp
        

    def losses_distribution(self):
        losses=np.empty(self.n)
        matrix=np.empty((len(self.EAD),self.n))
        for i in range(len(self.EAD)):
            matrix[i,:]=self.portfolio.multi_value(self.scenario,self.scenario_temp,self.n,self.t)

        for i in range(self.n):
            defaults = (matrix[:, i] <= self.PD).astype(int)
            losses[i]=self.EAD*self.LGD*defaults
