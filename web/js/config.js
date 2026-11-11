// Geometry of the simulated sorting line (world units are pixels; 400 px = 1 m).
export const CLASSES = ['plastic', 'glass', 'cardboard', 'metal', 'paper', 'organic', 'other'];
export const LABEL = {
  plastic: 'Plástico', glass: 'Vidrio', cardboard: 'Cartón', metal: 'Metal',
  paper: 'Papel', organic: 'Orgánico', other: 'Rechazo',
};
export const COLOR = {
  plastic: '#f2c14e', glass: '#3fc1b0', cardboard: '#c28f5c', metal: '#9db4d0',
  paper: '#ece7d6', organic: '#7ccf5f', other: '#8a8f99',
};

export const G = {
  PXM: 400,          // pixels per metre
  BELT_H: 480,       // belt width (1.2 m)
  BIN_H: 104,        // bins on each side of the belt
  CAM_X: 60,         // camera field of view starts here
  CAM_W: 640,
  GAP: 40,           // between camera and first station
  ST_W: 300,         // length of a robot station along the belt (0.75 m)
  N_ST: 3,
  TAIL: 80,
};
G.ST_X0 = G.CAM_X + G.CAM_W + G.GAP;
G.W = G.ST_X0 + G.N_ST * G.ST_W + G.TAIL;
G.H = G.BELT_H + 2 * G.BIN_H;
G.BELT_Y = G.BIN_H;  // belt top edge in canvas coordinates

// Each station sorts two fractions: [bin on the far side, bin on the near side].
export const STATIONS = [['plastic', 'metal'], ['paper', 'cardboard'], ['glass', 'organic']];

// Robot (gantry pick-and-place) kinematic limits and controller gains.
export const ARM = {
  VMAX: 1300,        // px/s  (3.25 m/s)
  AMAX: 14000,       // px/s^2 (35 m/s^2) used by the trajectory planner
  UMAX: 20000,       // actuator saturation for the feedback loop
  KP: 1600, KD: 80,  // PD gains: wn = 40 rad/s, zeta = 1
  T_PICK: 0.14,      // s the head follows the belt while the suction cup grips
  T_GRIP: 0.6,       // fraction of T_PICK at which the grip closes
  T_DROP: 0.08,
  MARGIN: 14,        // keep-out from the ends of the arm's zone
  PICK_TOL: 12,      // px: grips farther than this from the plan are reported
};

export const DEFAULTS = {
  beltSpeed: 0.30,   // m/s
  rate: 2.0,         // objects per second
  armsPerStation: 1,
  policy: 'edf',     // edf | nearest | fifo
  confThr: 0.50,
  gripFail: 0.03,
  perception: 'oracle',
  seed: 20261110,
};
