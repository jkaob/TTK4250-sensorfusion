"""

"""
# %% Imports

# types
from typing import (
    Tuple,
    List,
    TypeVar,
    Optional,
    Dict,
    Any,
    Union,
    Sequence,
    Generic,
    Iterable,
)
from mixturedata import MixtureParameters
from gaussparams import GaussParams
from estimatorduck import StateEstimator
from singledispatchmethod import singledispatchmethod

# packages
from dataclasses import dataclass

# from singledispatchmethod import singledispatchmethod
import numpy as np
from scipy import linalg
from scipy.special import logsumexp

# local
import discretebayes

# %% TypeVar and aliases
MT = TypeVar("MT")  # a type variable to be the mode type

# %% IMM
@dataclass
class IMM(Generic[MT]):
    # The M filters the IMM relies on
    filters: List[StateEstimator[MT]]
    # the transition matrix. PI[i, j] = probability of going from model i to j: shape (M, M)
    PI: np.ndarray
    
    # init mode probabilities if none is given
    initial_mode_probabilities: Optional[np.ndarray] = None

    def __post_init__(self):
        assert (
            self.PI.ndim == 2
        ), "Transition matrix PI shape must be (len(filters), len(filters))"
        assert (
            self.PI.shape[0] == self.PI.shape[1]
        ), "Transition matrix PI shape must be (len(filters), len(filters))"
        assert np.allclose(
            self.PI.sum(axis=1), 1
        ), "The rows of the transition matrix PI must sum to 1."

        assert (
            len(self.filters) == self.PI.shape[0]
        ), "Transition matrix PI shape must be (len(filters), len(filters))"


    def mix_probabilities(
        self,
        immstate: MixtureParameters[MT],
        Ts: float,
    ) -> Tuple[
        np.ndarray, np.ndarray
    ]:  # predicted_mode_probabilities, mix_probabilities: shapes = ((M, (M ,M))).
        """Calculate the predicted mode probability and the mixing probabilities."""
        
        predicted_mode_probabilities, mix_probabilities = discretebayes.discrete_bayes(immstate.weights, self.PI)
        
        # Optional assertions for debugging
        assert np.all(np.isfinite(predicted_mode_probabilities))
        assert np.all(np.isfinite(mix_probabilities))
        assert np.allclose(mix_probabilities.sum(axis=1), 1)

        return predicted_mode_probabilities, mix_probabilities        

    def mix_states(
        self,
        immstate: MixtureParameters[MT],
        # the mixing probabilities: shape=(M, M)
        mix_probabilities: np.ndarray,
    ) -> List[MT]:

        mixed_states = [fs.reduce_mixture(MixtureParameters(mix_prob_s, immstate.components)) for fs, mix_prob_s in zip(self.filters, mix_probabilities)]
       
        return mixed_states

    #KF prediction step for new mixed states
    def mode_matched_prediction(
        self,
        mode_states: List[MT],
        Ts: float,
    ) -> List[MT]:
        
        modestates_pred = [fs.predict(cs, Ts) for fs, cs in zip(self.filters, mode_states)]
        return modestates_pred

    ## Predict part of step 3
    def predict(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> MixtureParameters[MT]:
        """
        Predict the immstate Ts time units ahead approximating the mixture step.

        Ie. 
        Predict mode probabilities, 
        condition states on predicted mode, 
        appoximate resulting state distribution as Gaussian for each mode, 
        then predict each mode.
        """

        predicted_mode_probability, mixing_probability = self.mix_probabilities(immstate, Ts)
        mixed_mode_states: List[MT] = mix_states(immstate, mixing_probability)
        predicted_mode_states       = mode_matched_predicition(mixed_mode_states, Ts)
        predicted_immstate          = MixtureParameters(predicted_mode_probability, predicted_mode_states)
        
        return predicted_immstate

    ## update mean and cov, calculate measurement log likelihood: log(Delta_k^Sk)
    def mode_matched_update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> List[MT]:
        """Update each mode in immstate with z in sensor_state."""

        updated_state = [(fs.update(z, cs, sensor_state=sensor_state)) 
                         for fs, cs in zip(self.filters, immstate.components)]
        return updated_state

    def update_mode_probabilities(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Calculate the mode probabilities in immstate updated with z in sensor_state"""
        
        mode_loglikelihood =  np.array([
            fs.loglikelihood(z, cs, sensor_state=sensor_state) 
            for fs, cs in zip(self.filters, immstate.components)]) 

        logjoint = mode_loglikelihood + np.log(immstate.weights)
        updated_mode_probabilities = np.exp(logjoint - logsumexp(logjoint))

        assert np.all(
            np.isfinite(updated_mode_probabilities)
        ), "IMM.update_mode_probabilities: updated probabilities not finite "
        assert np.allclose(
            np.sum(updated_mode_probabilities), 1
        ), "IMM.update_mode_probabilities: updated probabilities does not sum to one"

        return updated_mode_probabilities

     ## Update part of step3
    def update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Update the immstate with z in sensor_state."""

        updated_weights = self.update_mode_probabilities(
            z, immstate, sensor_state=sensor_state)
        
        updated_states  = self.mode_matched_update(
            z, immstate, sensor_state=sensor_state)
        
        updated_immstate = MixtureParameters(updated_weights, updated_states)
        
        return updated_immstate

    def step(
        self,
        z,
        immstate: MixtureParameters[MT],
        Ts: float,
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Predict immstate with Ts time units followed by updating it with z in sensor_state"""
        
        predicted_immstate = self.predict(immstate, Ts) 
        updated_immstate   = self.update(z, predicted_immstate, sensor_state=sensor_state)

        return updated_immstate

    def loglikelihood(
        self,
        z: np.ndarray,
        immstate: MixtureParameters,
        *,
        sensor_state: Dict[str, Any] = None,
    ) -> float:
        
        raise NotImplementedError  # TODO: remove when implemented

        mode_conditioned_ll = np.fromiter(
            (
                None  # TODO: your state filter (fs under) should be able to calculate the mode conditional log likelihood at z from modestate_s
                for fs, modestate_s in zip(self.filters, immstate.components)
            ),
            dtype=float,
        )

        ll = None  # weighted average of likelihoods (not log!)

        assert np.isfinite(ll), "IMM.loglikelihood: ll not finite"
        assert isinstance(ll, float) or isinstance(
            ll.item(), float
        ), "IMM.loglikelihood: did not calculate ll to be a single float"
        return ll

    def reduce_mixture(
        self, immstate_mixture: MixtureParameters[MixtureParameters[MT]]
    ) -> MixtureParameters[MT]:
        """
        Approximate a mixture of immstates as a single immstate.

        We have Pr(a), Pr(s | a), p(x| s, a).
            - Pr(a) = immstate_mixture.weights
            - Pr(s | a=j) = immstate_mixture.components[j].weights
            - p(x | s=i, a=j) = immstate_mixture.components[j].components[i] # ie. Gaussian parameters

        So p(x, s) = sum_j Pr(a=j) Pr(s| a=j) p(x| s, a=j),
        which we want as a single probability Gaussian pair. Multiplying the above with
        1 = Pr(s)/Pr(s) and moving the denominator a little we have
        p(x, s) = Pr(s) sum_j [ Pr(a=j) Pr(s| a=j)/Pr(s) ]  p(x| s, a=j),
        where the bracketed term is Bayes for Pr(a=j|s). Thus the mode conditioned state estimate is.
        p(x | s) = sum_j Pr(a=j| s) p(x| s, a=j)

        That is:
            - we need to invoke discrete Bayes one time and
            - reduce self.filter[s].reduce_mixture for each s
        """

        raise NotImplementedError  # TODO remove this when done
        # extract probabilities as array
        ## eg. association weights/beta: Pr(a)
        weights = immstate_mixture.weights
        ## eg. the association conditioned mode probabilities element [j, s] is for association j and mode s: Pr(s | a = j)
        component_conditioned_mode_prob = np.array(
            [c.weights.ravel() for c in immstate_mixture.components]
        )

        # flip conditioning order with Bayes to get Pr(s), and Pr(a | s)
        mode_prob, mode_conditioned_component_prob = None  # TODO

        # We need to gather all the state parameters from the associations for mode s into a
        # single list in order to reduce it to a single parameter set.
        # for instance loop through the modes, gather the paramters for the association of this mode
        # into a single list and append the result of self.filters[s].reduce_mixture
        # The mode s for association j should be available as imm_mixture.components[j].components[s]

        mode_states: List[GaussParams] = None  # TODO

        immstate_reduced = MixtureParameters(mode_prob, mode_states)

        return immstate_reduced


    def estimate(self, immstate: MixtureParameters[MT]) -> GaussParams:
        """Calculate a state estimate with its covariance from immstate"""

        # ! You can assume all the modes have the same reduce and estimate function
        # ! and use eg. self.filters[0] functionality. # TODO
        state_reduced = self.filters[0].reduce_mixture(immstate)
        return self.filters[0].estimate(state_reduced)

    def gate(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        gate_size_square: float,
        sensor_state: Dict[str, Any] = None,
    ) -> bool:
        """Check if z is within the gate of any mode in immstate in sensor_state"""

        #  find which of the modes gates the measurement z
        mode_gated: List[bool] = [self.filters[0].gate(z, mode, gate_size_square, sensor_state=sensor_state) for mode in immstate.components]
            
        gated: bool = np.any(mode_gated) #  check if _any_ of the modes gated the measurement            
        return gated

    def NISes(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, np.ndarray]:
        """Calculate NIS per mode and the average"""
        NISes = np.array(
            [
                fs.NIS(z, ms, sensor_state=sensor_state)
                for fs, ms in zip(self.filters, immstate.components)
            ]
        )

        innovs = [
            fs.innovation(z, ms, sensor_state=sensor_state)
            for fs, ms in zip(self.filters, immstate.components)
        ]

        v_ave = np.average([gp.mean for gp in innovs], axis=0, weights=immstate.weights)
        S_ave = np.average([gp.cov for gp in innovs], axis=0, weights=immstate.weights)

        NIS = (v_ave * np.linalg.solve(S_ave, v_ave)).sum()
        return NIS, NISes

    def NEESes(
        self,
        immstate: MixtureParameters,
        x_true: np.ndarray,
        *,
        idx: Optional[Sequence[int]] = None,
    ):
        NEESes = np.array(
            [
                fs.NEES(ms, x_true, idx=idx)
                for fs, ms in zip(self.filters, immstate.components)
            ]
        )
        est = self.estimate(immstate)

        NEES = self.filters[0].NEES(est, x_true, idx=idx)  # HACK?
        return NEES, NEESes

""" 
    @singledispatchmethod
    def init_filter_state(
        self,
        init,  # Union[
        #     MixtureParameters, Dict[str, Any], Tuple[Sequence, Sequence], Sequence
        # ],
    ) -> MixtureParameters:
        
        Initialize the imm state to MixtureParameters.

        - If mode probabilities are not found they are initialized from self.initial_mode_probabilities.
        - If only one mode state is found, it is broadcasted to all modes.

        MixtureParameters: goes unaltered
        dict:
            ["weights", "probs", "probabilities", "mode_probs"]
                in this order can signify mode probabilities
            ["components", "modes"] signify the modes
        tuple: first element is mode probabilities and second is mode states
        Sequence: assumed to be only the mode states

        mode probabilities: array_like
        components:
        # TODO there are cases where MP unaltered can lead to trouble

        raise NotImplementedError(
            f"IMM do not know how to initialize a immstate from: {init}"
        )

    @init_filter_state.register
    def _(self, init: MixtureParameters[MT]) -> MixtureParameters[MT]:
        return init

    @init_filter_state.register(dict)
    def _(self, init: dict) -> MixtureParameters[MT]:
        # extract weights
        got_weights = False
        got_components = False
        for key in init:
            if not got_weights and key in [
                "weights",
                "probs",
                "probabilities",
                "mode_probs",
            ]:
                weights = np.asfarray([key])
                got_weights = True
            elif not got_components and key in ["components", "modes"]:
                components = self.init_components(init[key])
                got_components = True

        if not got_weights:
            weights = self.initial_mode_probabilities

        if not got_components:
            components = self.init_components(init)

        assert np.allclose(weights.sum(), 1), "Mode probabilities must sum to 1 for"

        return MixtureParameters(weights, components)

    @init_filter_state.register(tuple)
    def _(self, init: tuple) -> MixtureParameters[MT]:
        assert isinstance(init[0], Sized) and len(init[0]) == len(
            self.filters
        ), f"To initialize from tuple the first element must be of len(self.filters)={len(self.filters)}"

        weights = np.asfarray(init[0])
        components = self.init_compontents(init[1])
        return MixtureParameters(weights, components)

    @init_filter_state.register(Sequence)
    def _(self, init: Sequence) -> MixtureParameters[MT]:
        weights = self.initial_mode_probabilities
        components = self.init_components(init)
        return MixtureParameters(weights, components)

    @singledispatchmethod
    def init_components(self, init: "Union[Iterable, MT_like]") -> List[MT]:
        Make an instance or Iterable of the Mode Parameters into a list of mode parameters
        return [fs.init_filter_state(init) for fs in self.filters]

    @init_components.register(dict)
    def _(self, init: dict):
        return [fs.init_filter_state(init) for fs in self.filters]

    @init_components.register(Iterable)
    def _(self, init: Iterable) -> List[MT]:
        if isinstance(init[0], (np.ndarray, list)):
            return [
                fs.init_filter_state(init_s) for fs, init_s in zip(self.filters, init)
            ]
        else:
            return [fs.init_filter_state(init) for fs in self.filters]

    def estimate_sequence(
        self,
        # A sequence of measurements
        Z: Sequence[np.ndarray],
        # the initial KF state to use for either prediction or update (see start_with_prediction)
        init_immstate: MixtureParameters,
        # Time difference between Z's. If start_with_prediction: also diff before the first Z
        Ts: Union[float, Sequence[float]],
        *,
        # An optional sequence of the sensor states for when Z was recorded
        sensor_state: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
        # sets if Ts should be used for predicting before the first measurement in Z
        start_with_prediction: bool = False,
    ) -> Tuple[List[MixtureParameters], List[MixtureParameters], List[GaussParams]]:
        Create estimates for the whole time series of measurements.

        # sequence length
        K = len(Z)

        # Create and amend the sampling array
        Ts_start_idx = int(not start_with_prediction)
        Ts_arr = np.empty(K)
        Ts_arr[Ts_start_idx:] = Ts
        # Insert a zero time prediction for no prediction equivalence
        if not start_with_prediction:
            Ts_arr[0] = 0

        # Make sure the sensor_state_list actually is a sequence
        sensor_state_seq = sensor_state or [None] * K

        init_immstate = self.init_filter_state(init_immstate)

        immstate_upd = init_immstate

        immstate_pred_list = []
        immstate_upd_list = []
        estimates = []

        for z_k, Ts_k, ss_k in zip(Z, Ts_arr, sensor_state_seq):
            immstate_pred = self.predict(immstate_upd, Ts_k)
            immstate_upd = self.update(z_k, immstate_pred, sensor_state=ss_k)

            immstate_pred_list.append(immstate_pred)
            immstate_upd_list.append(immstate_upd)
            estimates.append(self.estimate(immstate_upd))

        return immstate_pred_list, immstate_upd_list, estimates
        """