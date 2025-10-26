"""Classes for storing known lineshapes and calibration information"""

from dataclasses import dataclass
from uncertainties import UFloat, ufloat_fromstr


@dataclass(eq=True, frozen=True)
class Line:
    wavelength: UFloat | None = None
    ion: str | None = None
    transition: tuple[str, str] | None = None  # lower, upper
    reference: str | None = None
    notes: str | None = None

    def __post_init__(self):
        if isinstance(self.wavelength, str):
            super().__setattr__('wavelength', ufloat_fromstr(self.wavelength))
        if isinstance(self.transition, str):
            if self.transition.count('-') != 1:
                raise ValueError(f'{self.transition} must either be a tuple of strings (lower, upper) or a string with one "-"')
            super().__setattr__('transistion', [level.strip() for level in self.transition.split('-')])


@dataclass
class Observation:
    """
    Stores information about specific observations of lines (typically calibration lines). Indepdenent_vars will
    typically have keys 'x' and possibly 't' describing the spectrometer position and time of the observation.

    Used by Calibrator to optimize line positions.
    """

    ind_vars: dict[str, UFloat | float]
    line: Line
    order: int = 1

    def __init__(self, ind_vars: dict[str, UFloat | float], line: Line, order: int = 1):
        self.ind_vars = ind_vars

        if not isinstance(line, Line):
            raise TypeError(f'line: {line} is not of type Line or Composite Line')

        self.line = line
        self.order = order

