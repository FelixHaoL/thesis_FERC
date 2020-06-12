import numpy as np
import pandas as pd
from scipy.optimize import newton

class Inverse_Weierstrass:
    '''
    initial paramers by calling XX = Inverse_Weierstrass(bind_width, spring_of_cantelever, KBT)
    '''

    def __init__(self, bin_width, k, kBT):  
        self.bin_width = bin_width
        self.k = k
        self.kBT = kBT
        self.Unfold_term = 0
        self.Refold_term = 0
        self.Fz_U = pd.DataFrame()
        self.Fz_weighted_histogram = pd.DataFrame()
        self.Fz_R = pd.DataFrame()
        self.W_U = pd.DataFrame()
        self.Wz_weighted_histogram = pd.DataFrame()
        self.Qz_weighted_histogram = pd.DataFrame()
        self.W_R = pd.DataFrame()
        self.delta_A = []
        self.logarithm = 0

    """
        These two functions below convert force-extension data to force-z data which needed for Inverse_Weierstrass using
                                             Extension = Z - Force/k
    And it also rename the DataFrame columns to fit the further algorithm 

    
        DataFrame:It should contain all unfold(refold) trajectories, and each has two columns names 
    'Force_column_name'+'n' for force columns and 'Ext_column_name'+'n' for extension columns.
    I propose names like these Force_smthU0 and ext_smthR0 for Force channal of 1st unfolding trajectory and 
    extension channal of 1st refolding trajectory respectively 

        direction:'U' or 'R' for unfold and refold respectively.

        Force_column_name and Ext_column_name:The column names of DataFrame except the number.

        trajectorie_n:Number of trajectories the DataFrame contains
    """
    def convert_FEC_to_FZC(self, DataFrame):  # first column be force data and second's be extension
        # Extension = Z - Force/k
        Z_Series = DataFrame.iloc[:, 1] + DataFrame.iloc[:, 0] / self.k
        new_DataFrame = pd.DataFrame({'Force': DataFrame.iloc[:, 0],
                                      'Z': Z_Series})
        return new_DataFrame

    def convert_FEC_to_FZC_all(self, DataFrame, direction, Force_column_name, Ext_column_name, trajectorie_n):
        FZ = pd.DataFrame()
        for i in range(trajectorie_n):
            new_col = ['Force' + str(direction) + str(i), 'Z' + str(direction) + str(i)]
            df = self.convert_FEC_to_FZC(DataFrame[[Force_column_name + str(i), Ext_column_name + str(i)]])
            FZ[new_col[0]] = df.iloc[:, 0]
            FZ[new_col[1]] = df.iloc[:, 1]
        return FZ



    """
    The 3 functions below Bin the data into force_Z bins and Use composite trapezoidal rule to compute the area under trajectories 
    """
    def calculate_meanFz(self, DataFrame):  # first column be force data and second's be Z
        Fz = []
        for z in np.arange(self.lowerbound_z, self.upperbound_z, self.bin_width):
            Fz.append(DataFrame.loc[(DataFrame[DataFrame.columns[1]] >= z) &
                                    (DataFrame[DataFrame.columns[1]] < (z + self.bin_width)),
                                    DataFrame.columns[0]].mean())
        return Fz

    def calculate_Wz_unfold(self, Fz):  # first column be force data and second's be Z
        Wz = []
        for i in range(1,len(Fz)+1):
            Wz.append(np.trapz(Fz[:i], np.arange(self.lowerbound_z, self.upperbound_z, self.bin_width)[:i]))
        return Wz[:]

    def calculate_Wz_refold(self, Fz):  # first column be force data and second's be Z
        Wz = []
        for i in range(len(Fz)-1,-1,-1):
            Wz.append(-np.trapz(Fz[i:], np.arange(self.lowerbound_z, self.upperbound_z, self.bin_width)[i:]))
        return Wz[::-1]

        """
        Jarsynski formular to calculate Az and G0 by calling the functions 'calculate_Az_Jarsynski()' and 'Jarsynski_Gq()' successively;
        The first function returns the Az series and the latter returns a DataFrame with G0 and q(ext) columns
        """
    def calculate_Az_Jarsynski(self, df_FZ, lowerbound_z, upperbound_z):
        self.lowerbound_z = lowerbound_z
        self.upperbound_z = upperbound_z
        self.n_U = len(df_FZ.columns) / 2

        for i in range(int(self.n_U)):
            direction = 'U'
            self.Fz_U['traj' + str(i)] = self.calculate_meanFz(df_FZ[['Force' + str(direction) + str(i),
                                                                      'Z' + str(direction) + str(i)]])
            self.W_U['traj' + str(i)] = self.calculate_Wz_unfold(self.Fz_U['traj' + str(i)])

        self.WZexp = np.exp((-1/self.kBT)*self.W_U)
        self.Az = np.log(self.WZexp.mean(axis=1))*-self.kBT
        return self.Az

    def work_weighted_derivatives(self):
        work_weighted_force = [self.Fz_U.iloc[:,i]*self.WZexp.iloc[:,i] for i in range(len(self.Fz_U.columns))]
        work_weighted_force = pd.DataFrame(np.transpose(work_weighted_force))
        work_weighted_force = work_weighted_force.mean(axis=1) / self.WZexp.mean(axis=1)


        work_weighted_force_sqrt = [(self.Fz_U.iloc[:,i]**2)*self.WZexp.iloc[:,i] for i in range(len(self.Fz_U.columns))]
        work_weighted_force_sqrt = pd.DataFrame(np.transpose(work_weighted_force_sqrt))
        work_weighted_force_sqrt = work_weighted_force_sqrt.mean(axis=1) / self.WZexp.mean(axis=1)

        self.work_weighted_f = work_weighted_force
        self.work_weighted_f2 = work_weighted_force_sqrt
        return work_weighted_force, work_weighted_force_sqrt

    def Jarsynski_Gq(self):
        f1, f2 = self.work_weighted_derivatives()
        self.logarithm = f2 - f1**2
        self.logarithm = self.logarithm / (self.k*self.kBT)
        
        z = np.arange(self.lowerbound_z, self.upperbound_z, self.bin_width) + self.bin_width / 2
        self.q = z - f1 / self.k
        self.first_deri_term = f1 ** 2 / (2 * self.k)
        self.second_deri_term = 0.5 * self.kBT * np.log(self.logarithm)
        self.Gq = self.Az - self.first_deri_term + self.second_deri_term
        result = pd.DataFrame({'Gq': self.Gq,
                               'q': self.q})
        return result

        """
        Minh_Adib_formular to calculate Az and G0 by calling the functions 'calculate_Az_MinhAdib' and 'Calculate_Gq()' successively;
        'calculate_Az_MinhAdib' returns the Az series and 'Calculate_Gq()' returns a DataFrame with G0 and ext columns
        """
        
    def Bennett_acceptance_ratio_formular(self, delta_A, Wz_unfold_list, Wz_refold_list):
        formular_left = np.mean([(self.n_R + self.n_U * np.exp(self.kBT ** -1 * (value - delta_A))) ** -1 for value in Wz_unfold_list])

        formular_righ = np.mean([(self.n_U + self.n_R * np.exp(self.kBT ** -1 * (value + delta_A))) ** -1 for value in Wz_refold_list])
        
        return formular_left - formular_righ

    def Minh_Adib_formular_SumRightTerms(self):
        forward_numerator = self.n_U * np.exp((-1 / self.kBT) * self.W_U)
        forward_denominator = self.n_U + self.n_R * np.exp((-1 / self.kBT) * (self.W_U.iloc[-1] - self.delta_A))
        forward_mean = np.mean((forward_numerator / forward_denominator), axis=1)

        backward_numerator = self.n_R * np.exp((-1 / self.kBT) * (self.W_R + self.delta_A))
        backward_denominator = self.n_R + self.n_U * np.exp((-1 / self.kBT) * (self.W_R.iloc[0] + self.delta_A))
        backward_mean = np.mean((backward_numerator / backward_denominator), axis=1)

        return forward_mean + backward_mean

    def calculate_Az_MinhAdib(self, df_U, df_R, lowerbound_z, upperbound_z):
        """
        Newton–Raphson method solving ΔA in Bennett acceptance ratio formular
        here ΔA means the area under curve from lower_z (z0) to upper_z (z1)
        The initial guess of ΔA can lead to wrong result if guess is too far away
        Here I use the corresponding unfold Wz(z1) to initiate the ΔA guesses
        """
        self.lowerbound_z = lowerbound_z
        self.upperbound_z = upperbound_z
        self.n_U = len(df_U.columns) / 2
        self.n_R = len(df_R.columns) / 2
        assert (self.lowerbound_z >= 0) & (self.upperbound_z >= 0), 'Did you forget to assigh Z range or set it wrong(z<0)?'

        for i in range(int(self.n_U)):
            direction = 'U'
            self.Fz_U['traj' + str(i)] = self.calculate_meanFz(df_U[['Force' + str(direction) + str(i),
                                                                     'Z' + str(direction) + str(i)]])
            self.W_U['traj' + str(i)] = self.calculate_Wz_unfold(self.Fz_U['traj' + str(i)])
        for i in range(int(self.n_R)):
            direction = 'R'
            self.Fz_R['traj' + str(i)] = self.calculate_meanFz(df_R[['Force' + str(direction) + str(i),
                                                                     'Z' + str(direction) + str(i)]])
            self.W_R['traj' + str(i)] = self.calculate_Wz_refold(self.Fz_R['traj' + str(i)])

        
        self.delta_A = newton(self.Bennett_acceptance_ratio_formular, np.mean(self.W_U.iloc[-1].tolist()),
                              args=(self.W_U.iloc[-1].tolist(), self.W_R.iloc[0].tolist()), maxiter=500)
        self.Az = -self.kBT * np.log(self.Minh_Adib_formular_SumRightTerms())

        return self.Az

    def Calculate_Gq(self, Az):
        '''
        numerically compute the derivatives of Az
        '''
        self.Az_prime = np.gradient(Az,self.bin_width)
        Az_prime2 = np.gradient(self.Az_prime,self.bin_width)
        z = np.arange(self.lowerbound_z, self.upperbound_z, self.bin_width) + self.bin_width / 2
        self.q = z - self.Az_prime / self.k
        self.first_deri_term = self.Az_prime ** 2 / (2 * self.k)
        self.second_deri_term = 0.5 * self.kBT * np.log(1 - Az_prime2 / self.k)
        self.Gq = Az - self.first_deri_term + self.second_deri_term
        result = pd.DataFrame({'Gq': self.Gq,
                               'q': self.q})
        return result


    '''
    weighted histogram method
    '''


    def energy_cantilever(self, Fz):
        W_cantilever = Fz**2 / (2 * self.k)
        return W_cantilever
    

    def calculate_Wz_weighted_histogram(self, df_FZ, lowerbound_z, upperbound_z):
        self.lowerbound_z = lowerbound_z
        self.upperbound_z = upperbound_z
        self.n_U = len(df_FZ.columns) / 2

        for i in range(int(self.n_U)):
            FZ_temp = df_FZ[['ForceU'+ str(i),'ZU' + str(i)]]
            self.Fz_weighted_histogram['traj' + str(i)] = self.calculate_meanFz(FZ_temp) 
            #self.Wz_weighted_histogram['traj' + str(i)] = self.calculate_Wz_unfold(self.Fz_weighted_histogram['traj' + str(i)]) - self.energy_cantilever(self.Fz_weighted_histogram['traj' + str(i)][0])
            self.Wz_weighted_histogram['traj' + str(i)] = self.calculate_Wz_unfold(self.Fz_weighted_histogram['traj' + str(i)])
        return self.Wz_weighted_histogram

    def calculate_Qz_weighted_histogram(self, df_FZ, lowerbound_z, upperbound_z):
        self.lowerbound_z = lowerbound_z
        self.upperbound_z = upperbound_z
        self.n_U = len(df_FZ.columns) / 2

        for i in range(int(self.n_U)):
            FZ_temp = df_FZ[['ForceU'+ str(i),'ZU' + str(i)]]
            Qz_temp = pd.DataFrame({'extU': FZ_temp['ZU'+ str(i)] - FZ_temp['ForceU'+ str(i)]/self.k, 'ZU':FZ_temp['ZU'+ str(i)]})
            self.Qz_weighted_histogram['traj' + str(i)] = self.calculate_meanFz(Qz_temp)
        
        return self.Qz_weighted_histogram



    def G_weighted_histogram(self, Wz_df):
        alpha = np.mean(np.exp(Wz_df * (-1/ self.kBT)), axis=1)
        beta = np.exp((-1/ self.kBT) * self.energy_cantilever(self.Fz_weighted_histogram)).mean(axis=1)
        weight = alpha.iloc[-1]
        G = np.log((alpha/weight)/(np.sum(beta/weight))) * (-self.kBT)

        z = np.arange(self.lowerbound_z, self.upperbound_z, self.bin_width)
        q = z - (self.Fz_weighted_histogram.mean(axis=1) / self.k)

        result = pd.DataFrame({'G0':G, 'ext':q})
        return result


