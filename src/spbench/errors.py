"""Errors that keep invalid runs, missing backends, and bad plots distinct."""


class BenchmarkError(Exception):
    """Base class for expected benchmark failures."""


class InvalidConfiguration(BenchmarkError):
    """The requested case is not a legal configuration of a required method."""


class UnavailableBackend(BenchmarkError):
    """A legal case cannot run because hardware or an upstream package is missing.

    This is not permission to replace the method with dense attention or with a
    different distributed algorithm.
    """


class PlotDataError(BenchmarkError):
    """A figure would have mixed fixtures, hypotheses, or failed runs into measured data."""
