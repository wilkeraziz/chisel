
        # estimate (normalised) importance weights
        """
        self.log_unnorm_r_ = np.array([np.sumraw.log_ur + np.log(raw.count) for Dy in self.support_], float)
        log_Zr = np.logaddexp.reduce(log_ur)
        log_r = log_ur - log_Zr
        importance = np.exp(log_r)
        """

        # unnormalised p(y)
        #   Zp * p(y) = \sum_d \gamma_y(d) * ur(d)
        # where ur(d) is the unnormalised importance weight of d
        self.unnorm_p_ = np.array([np.sum(d.importance * d.count for d in Dy) for Dy in self.support_], float)
        #HACK self.unnorm_p_ = np.array([self.r_[i] for i, Dy in enumerate(self.support_)], float)
        ###print >> sys.stderr, 'unp(y)=', self.unnorm_p_
        # p's normalising constant
        self.Zp_ = self.unnorm_p_.sum(0)
        
        # p(y) = unnorm_p(y) / Zp
        self.Py_ = self.unnorm_p_ / self.Zp_
        ###print >> sys.stderr, 'p(y)=', self.Py_


        # unnormalised expected f(y) -- feature vector wrt the target distribution p
        #   Z <f_y> = \sum_{d \in Dy} gamma_y(d) f(d) ur(d) n(d)
        self.unnorm_f_ = np.array([reduce(sum, (d.vector.as_array(p_features) * d.importance * d.count for d in Dy)) for Dy in self.support_], float)
        # normalised expected f(y)
        #   <f(y)> = unnorm_f(y)/unnorm_p(y)
        self.Fy_ = np.array([self.unnorm_f_[i]/self.unnorm_p_[i] for i, Dy in enumerate(self.support_)])
        # expected feature vector <f(d)>
        self.uf_ = self.unnorm_f_.sum(0) / self.Zp_

        # unnoralised expected g(y) -- feature vector wrt the instrumental distribution q
        #   Z <g_y> = \sum_{d \in Dy} gamma_y(d) g(d) ur(d) n(d)
        self.unnorm_g_ = np.array([reduce(sum, (d.vector.as_array(q_features) * d.importance * d.count for d in Dy)) for Dy in self.support_], float)
        # normalised expected g(y)
        self.Gy_ = np.array([self.unnorm_g_[i]/self.unnorm_p_[i] for i, Dy in enumerate(self.support_)])
    #def qq(self, i, normalise=True):
    #    return self.unnormalised_q_[i] if not normalise else self.normalised_q_[i]

        # expected feature vector <g(d)>
        self.ug_ = self.unnorm_g_.sum(0) / self.Zp_
        
        # TODO: revise all estimates
        # normalised p(y) 
        #self.Py_ = np.array([np.sum(d.importance for d in Dy) for Dy in self.support_], float)
