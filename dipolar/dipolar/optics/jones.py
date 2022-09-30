import numpy as np

def polarization(angle=0, phase=0):
    # convert degree to radiants
    angle = np.deg2rad(angle)
    phase = np.deg2rad(phase)

    # calculate the polarization state
    P = np.array([np.cos(angle), np.sin(angle) * np.exp(-1j * phase)])

    return P

def polarizer(angle=0):
    # convert degree to radiants
    angle = np.deg2rad(angle)

    # calculate the polarizer jones matrix
    M = np.array([[np.cos(angle) ** 2, np.cos(angle) * np.sin(angle)],
                  [np.cos(angle) * np.sin(angle), np.sin(angle) ** 2]])

    return M

def phase_plate(phase=0, rotation=0):
    # convert degree to radiants
    phase = np.deg2rad(phase)
    rotation = np.deg2rad(rotation)

    # calculate the phase plate matrix
    M = np.array([[1, 0],
                  [0, np.exp(-1j * phase)]])

    # rotate the phase plate if required
    if rotation != 0:
        R = lambda x: np.array([[np.cos(x), np.sin(x)],
                                [-np.sin(x), np.cos(x)]])
        M = R(-rotation) @ M @ R(rotation)

    return M

