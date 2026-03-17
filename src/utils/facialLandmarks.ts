import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export interface LandmarkPoint {
  x: number;
  y: number;
  z?: number;
}

export interface FacialMetrics {
  ear: number;
  mar: number;
  drowsinessLevel: 'safe' | 'warning' | 'danger';
}

// Eye landmarks indices for EAR calculation
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];

// Mouth landmarks indices for MAR calculation
const MOUTH = [61, 291, 39, 181, 0, 17, 269, 405];

export const calculateDistance = (p1: LandmarkPoint, p2: LandmarkPoint): number => {
  return Math.sqrt(
    Math.pow(p2.x - p1.x, 2) + 
    Math.pow(p2.y - p1.y, 2)
  );
};

export const calculateEAR = (landmarks: LandmarkPoint[]): number => {
  const calculateEyeEAR = (eyeIndices: number[]): number => {
    const vertical1 = calculateDistance(landmarks[eyeIndices[1]], landmarks[eyeIndices[5]]);
    const vertical2 = calculateDistance(landmarks[eyeIndices[2]], landmarks[eyeIndices[4]]);
    const horizontal = calculateDistance(landmarks[eyeIndices[0]], landmarks[eyeIndices[3]]);
    
    return (vertical1 + vertical2) / (2.0 * horizontal);
  };

  const leftEAR = calculateEyeEAR(LEFT_EYE);
  const rightEAR = calculateEyeEAR(RIGHT_EYE);
  
  return (leftEAR + rightEAR) / 2.0;
};

export const calculateMAR = (landmarks: LandmarkPoint[]): number => {
  const vertical1 = calculateDistance(landmarks[MOUTH[1]], landmarks[MOUTH[7]]);
  const vertical2 = calculateDistance(landmarks[MOUTH[2]], landmarks[MOUTH[6]]);
  const vertical3 = calculateDistance(landmarks[MOUTH[3]], landmarks[MOUTH[5]]);
  const horizontal = calculateDistance(landmarks[MOUTH[0]], landmarks[MOUTH[4]]);
  
  return (vertical1 + vertical2 + vertical3) / (3.0 * horizontal);
};

export const getDrowsinessLevel = (ear: number, mar: number): 'safe' | 'warning' | 'danger' => {
  const EAR_THRESHOLD_DANGER = 0.15;
  const EAR_THRESHOLD_WARNING = 0.20;
  const MAR_THRESHOLD = 0.65;
  
  if (ear < EAR_THRESHOLD_DANGER || mar > MAR_THRESHOLD) {
    return 'danger';
  } else if (ear < EAR_THRESHOLD_WARNING) {
    return 'warning';
  }
  
  return 'safe';
};

let faceLandmarker: FaceLandmarker | null = null;

export const initializeFaceLandmarker = async (): Promise<FaceLandmarker> => {
  if (faceLandmarker) return faceLandmarker;

  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 1
  });

  return faceLandmarker;
};
