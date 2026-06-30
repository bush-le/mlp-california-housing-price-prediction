import numpy as np

class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
    def _mvn_logpdf(self, X, mean, cov):
        n_features = X.shape[1]
        cov = cov + np.eye(n_features) * 1e-4
        diff = X - mean
        inv_cov = np.linalg.inv(cov)
        exponent = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        sign, logdet = np.linalg.slogdet(cov)
        return -0.5 * (n_features * np.log(2 * np.pi) + logdet + exponent)
        
    def fit(self, X, log_fn=None):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices]
        self.covariances_ = np.array([np.cov(X.T) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        log_likelihoods = []
        for i in range(self.max_iter):
            # E-step
            log_resp = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                log_resp[:, k] = np.log(self.weights_[k] + 1e-12) + self._mvn_logpdf(X, self.means_[k], self.covariances_[k])
                
            max_log_resp = np.max(log_resp, axis=1, keepdims=True)
            log_prob_norm = max_log_resp + np.log(np.sum(np.exp(log_resp - max_log_resp), axis=1, keepdims=True))
            
            log_likelihood = np.sum(log_prob_norm)
            log_likelihoods.append(log_likelihood)
            
            responsibilities = np.exp(log_resp - log_prob_norm)
            
            # M-step
            N_k = np.sum(responsibilities, axis=0)
            for k in range(self.n_components):
                self.means_[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]
                diff = X - self.means_[k]
                self.covariances_[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k]
                self.weights_[k] = N_k[k] / n_samples
                
            if log_fn:
                log_fn(f"  EM Iteration {i+1}: log-likelihood = {log_likelihood:.4f}")
                
            if i > 0 and np.abs(log_likelihood - log_likelihoods[-2]) < self.tol:
                if log_fn:
                    log_fn(f"  Converged after {i+1} iterations.")
                break
                
        self.log_likelihood_ = log_likelihoods[-1]
        self.history_ = log_likelihoods
        return self
        
    def score(self, X):
        log_resp = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            log_resp[:, k] = np.log(self.weights_[k] + 1e-12) + self._mvn_logpdf(X, self.means_[k], self.covariances_[k])
        max_log_resp = np.max(log_resp, axis=1, keepdims=True)
        log_prob_norm = max_log_resp + np.log(np.sum(np.exp(log_resp - max_log_resp), axis=1, keepdims=True))
        return np.sum(log_prob_norm)