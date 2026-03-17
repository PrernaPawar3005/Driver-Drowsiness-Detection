import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Play, Square, AlertTriangle, Eye, Smile } from 'lucide-react';
import { initializeFaceLandmarker, calculateEAR, calculateMAR, getDrowsinessLevel, type FacialMetrics } from '@/utils/facialLandmarks';
import { toast } from 'sonner';
import type { FaceLandmarker } from '@mediapipe/tasks-vision';

interface AlertLog {
  timestamp: Date;
  level: 'warning' | 'danger';
  message: string;
}

const DrowsinessDetector = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [metrics, setMetrics] = useState<FacialMetrics>({ ear: 0, mar: 0, drowsinessLevel: 'safe' });
  const [alertLogs, setAlertLogs] = useState<AlertLog[]>([]);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationFrameRef = useRef<number>();
  const lastAlertTimeRef = useRef<number>(0);
  const dangerCountRef = useRef<number>(0);
  const consecutiveFramesRef = useRef<{ danger: number; warning: number; safe: number }>({
    danger: 0,
    warning: 0,
    safe: 0
  });

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      stopWebcam();
    };
  }, []);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (error) {
      console.error('Error accessing webcam:', error);
      toast.error('Failed to access webcam. Please grant camera permissions.');
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const addAlertLog = (level: 'warning' | 'danger', message: string) => {
    const now = Date.now();
    if (now - lastAlertTimeRef.current > 3000) {
      setAlertLogs(prev => [{
        timestamp: new Date(),
        level,
        message
      }, ...prev.slice(0, 9)]);
      lastAlertTimeRef.current = now;
      
      if (level === 'danger') {
        toast.error(message, { duration: 3000 });
      } else {
        toast.warning(message, { duration: 2000 });
      }
    }
  };

  const detectFacialLandmarks = async () => {
    if (!videoRef.current || !canvasRef.current || !faceLandmarkerRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx || video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(detectFacialLandmarks);
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const startTimeMs = performance.now();
    const results = faceLandmarkerRef.current.detectForVideo(video, startTimeMs);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      const landmarks = results.faceLandmarks[0];
      
      // Draw landmarks
      ctx.fillStyle = 'rgba(0, 255, 255, 0.5)';
      landmarks.forEach((landmark) => {
        ctx.beginPath();
        ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 1, 0, 2 * Math.PI);
        ctx.fill();
      });

      // Calculate metrics
      const ear = calculateEAR(landmarks);
      const mar = calculateMAR(landmarks);
      const drowsinessLevel = getDrowsinessLevel(ear, mar);

      setMetrics({ ear, mar, drowsinessLevel });

      // Track consecutive frames for each state
      if (drowsinessLevel === 'danger') {
        consecutiveFramesRef.current.danger++;
        consecutiveFramesRef.current.warning = 0;
        consecutiveFramesRef.current.safe = 0;
        
        // Only alert if eyes are closed for sustained period (20+ frames = ~0.7 seconds)
        if (consecutiveFramesRef.current.danger > 20) {
          dangerCountRef.current++;
          if (dangerCountRef.current > 3) {
            addAlertLog('danger', '⚠️ DROWSINESS DETECTED! Take a break immediately!');
          }
        }
      } else if (drowsinessLevel === 'warning') {
        consecutiveFramesRef.current.warning++;
        consecutiveFramesRef.current.danger = 0;
        consecutiveFramesRef.current.safe = 0;
        
        // Only warn if sustained (15+ frames = ~0.5 seconds)
        if (consecutiveFramesRef.current.warning > 15 && dangerCountRef.current > 2) {
          addAlertLog('warning', '⚡ Warning: Signs of fatigue detected');
        }
        dangerCountRef.current = Math.max(0, dangerCountRef.current - 1);
      } else {
        consecutiveFramesRef.current.safe++;
        consecutiveFramesRef.current.danger = 0;
        consecutiveFramesRef.current.warning = 0;
        
        // Reset danger count faster when eyes are open
        if (consecutiveFramesRef.current.safe > 10) {
          dangerCountRef.current = Math.max(0, dangerCountRef.current - 2);
        }
      }
    }

    animationFrameRef.current = requestAnimationFrame(detectFacialLandmarks);
  };

  const handleStartDetection = async () => {
    try {
      toast.info('Initializing face detection...');
      faceLandmarkerRef.current = await initializeFaceLandmarker();
      await startWebcam();
      dangerCountRef.current = 0;
      setIsDetecting(true);
      
      // Start detection loop after a short delay to ensure video is ready
      setTimeout(() => {
        console.log('Starting detection loop');
        detectFacialLandmarks();
        toast.success('Detection started');
      }, 500);
    } catch (error) {
      console.error('Error starting detection:', error);
      toast.error('Failed to initialize face detection');
    }
  };

  const handleStopDetection = () => {
    setIsDetecting(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    stopWebcam();
    dangerCountRef.current = 0;
    consecutiveFramesRef.current = { danger: 0, warning: 0, safe: 0 };
    toast.info('Detection stopped');
  };

  const getStatusColor = () => {
    switch (metrics.drowsinessLevel) {
      case 'danger': return 'border-danger shadow-danger';
      case 'warning': return 'border-warning';
      case 'safe': return 'border-safe shadow-safe';
      default: return '';
    }
  };

  const getStatusBg = () => {
    switch (metrics.drowsinessLevel) {
      case 'danger': return 'bg-gradient-danger';
      case 'warning': return 'bg-warning/20';
      case 'safe': return 'bg-gradient-safe';
      default: return '';
    }
  };

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-primary bg-clip-text text-transparent">
            Driver Drowsiness Detection
          </h1>
          <p className="text-muted-foreground text-lg">
            AI-powered facial landmark analysis for real-time drowsiness monitoring
          </p>
        </div>

        {/* Status Banner */}
        <Card className={`p-6 transition-all duration-300 ${getStatusColor()}`}>
          <div className={`flex items-center justify-center gap-4 p-4 rounded-lg ${getStatusBg()}`}>
            <AlertTriangle className={`w-8 h-8 ${
              metrics.drowsinessLevel === 'danger' ? 'text-danger animate-pulse' :
              metrics.drowsinessLevel === 'warning' ? 'text-warning' :
              'text-safe'
            }`} />
            <div className="text-center">
              <div className="text-2xl font-bold uppercase tracking-wider">
                {metrics.drowsinessLevel === 'danger' ? '⚠️ DROWSINESS ALERT' :
                 metrics.drowsinessLevel === 'warning' ? '⚡ FATIGUE WARNING' :
                 '✓ DRIVER ALERT'}
              </div>
            </div>
          </div>
        </Card>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Video Feed */}
          <div className="lg:col-span-2 space-y-4">
            <Card className="p-6">
              <div className="relative aspect-video bg-secondary rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  className="absolute inset-0 w-full h-full object-cover hidden"
                  playsInline
                />
                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 w-full h-full object-cover"
                />
                {!isDetecting && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center space-y-4">
                      <Eye className="w-16 h-16 mx-auto text-primary" />
                      <p className="text-muted-foreground">Click Start to begin detection</p>
                    </div>
                  </div>
                )}
              </div>
              
              <div className="flex gap-4 mt-4">
                {!isDetecting ? (
                  <Button 
                    onClick={handleStartDetection}
                    className="flex-1 bg-gradient-primary hover:opacity-90"
                    size="lg"
                  >
                    <Play className="w-5 h-5 mr-2" />
                    Start Detection
                  </Button>
                ) : (
                  <Button 
                    onClick={handleStopDetection}
                    variant="destructive"
                    className="flex-1"
                    size="lg"
                  >
                    <Square className="w-5 h-5 mr-2" />
                    Stop Detection
                  </Button>
                )}
              </div>
            </Card>
          </div>

          {/* Metrics & Logs */}
          <div className="space-y-6">
            {/* Metrics */}
            <Card className="p-6 space-y-4">
              <h3 className="text-xl font-bold">Real-time Metrics</h3>
              
              <div className="space-y-3">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Eye className="w-5 h-5 text-primary" />
                      <span className="font-medium">Eye Aspect Ratio</span>
                    </div>
                    <span className="text-2xl font-mono font-bold">
                      {metrics.ear.toFixed(3)}
                    </span>
                  </div>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-300 ${
                        metrics.drowsinessLevel === 'danger' ? 'bg-danger' :
                        metrics.drowsinessLevel === 'warning' ? 'bg-warning' :
                        'bg-safe'
                      }`}
                      style={{ width: `${Math.min(metrics.ear * 200, 100)}%` }}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Smile className="w-5 h-5 text-primary" />
                      <span className="font-medium">Mouth Aspect Ratio</span>
                    </div>
                    <span className="text-2xl font-mono font-bold">
                      {metrics.mar.toFixed(3)}
                    </span>
                  </div>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-300 ${
                        metrics.mar > 0.6 ? 'bg-danger' : 'bg-primary'
                      }`}
                      style={{ width: `${Math.min(metrics.mar * 100, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            </Card>

            {/* Alert History */}
            <Card className="p-6">
              <h3 className="text-xl font-bold mb-4">Alert History</h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {alertLogs.length === 0 ? (
                  <p className="text-muted-foreground text-sm text-center py-8">
                    No alerts yet
                  </p>
                ) : (
                  alertLogs.map((log, idx) => (
                    <div 
                      key={idx}
                      className={`p-3 rounded-lg border ${
                        log.level === 'danger' 
                          ? 'border-danger bg-danger/10' 
                          : 'border-warning bg-warning/10'
                      }`}
                    >
                      <div className="text-xs text-muted-foreground mb-1">
                        {log.timestamp.toLocaleTimeString()}
                      </div>
                      <div className="text-sm">{log.message}</div>
                    </div>
                  ))
                )}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DrowsinessDetector;
