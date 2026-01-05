'use client';

import React, { useState } from 'react';
import { Button } from '@/components/livekit/button';

interface ProctoringSummary {
  success: boolean;
  report_text?: string;
  report_file?: string;
  video_url?: string;
  video_path?: string;
  integrity_score?: number;
  left_gaze_duration?: number;
  right_gaze_duration?: number;
  multiple_face_periods?: number;
  warnings?: string[];
  duration?: number;
  error?: string;
}

export function VideoProctoring() {
  const [videoUrl, setVideoUrl] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<ProctoringSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';

  // Auto-fill video URL from query parameter
  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const urlParam = params.get('video_url');
      if (urlParam) {
        setVideoUrl(urlParam);
      }
    }
  }, []);

  const analyzeVideo = async () => {
    if (!videoUrl.trim()) {
      setError('Please enter a video URL');
      return;
    }

    setAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/proctoring/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ video_url: videoUrl }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Analysis failed');
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze video');
      console.error('Error analyzing video:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const getIntegrityColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Video Proctoring Analysis</h1>
        <p className="text-muted-foreground">
          Analyze interview videos for gaze direction and multiple face detection
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-card rounded-lg border p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Analyze Video</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Video URL (Azure Blob Storage)
            </label>
            <input
              type="text"
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
              placeholder="https://storage.blob.core.windows.net/..."
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
              disabled={analyzing}
            />
          </div>

          <Button
            onClick={analyzeVideo}
            disabled={analyzing || !videoUrl.trim()}
            variant="primary"
            size="lg"
            className="w-full"
          >
            {analyzing ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                Analyzing Video...
              </>
            ) : (
              'Analyze Video'
            )}
          </Button>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              <p className="font-medium">Error:</p>
              <p>{error}</p>
            </div>
          )}
        </div>
      </div>

      {/* Results Section */}
      {result && result.success && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-card rounded-lg border p-4">
              <p className="text-sm text-muted-foreground mb-1">Integrity Score</p>
              <p className={`text-3xl font-bold ${getIntegrityColor(result.integrity_score || 0)}`}>
                {result.integrity_score?.toFixed(1)}%
              </p>
            </div>

            <div className="bg-card rounded-lg border p-4">
              <p className="text-sm text-muted-foreground mb-1">Duration</p>
              <p className="text-3xl font-bold">
                {result.duration ? `${(result.duration / 60).toFixed(1)}m` : 'N/A'}
              </p>
            </div>

            <div className="bg-card rounded-lg border p-4">
              <p className="text-sm text-muted-foreground mb-1">Looking Away</p>
              <p className="text-3xl font-bold">
                {((result.left_gaze_duration || 0) + (result.right_gaze_duration || 0)).toFixed(1)}s
              </p>
            </div>

            <div className="bg-card rounded-lg border p-4">
              <p className="text-sm text-muted-foreground mb-1">Multiple Faces</p>
              <p className="text-3xl font-bold">
                {result.multiple_face_periods || 0}
              </p>
            </div>
          </div>

          {/* Warnings */}
          {result.warnings && result.warnings.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h3 className="font-semibold text-yellow-900 mb-2">⚠️ Warnings</h3>
              <ul className="space-y-1">
                {result.warnings.map((warning, idx) => (
                  <li key={idx} className="text-yellow-800">{warning}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Video Player */}
          <div className="bg-card rounded-lg border p-6">
            <h3 className="text-xl font-semibold mb-4">Interview Video</h3>
            <div className="aspect-video bg-black rounded-lg overflow-hidden">
              <video
                key={result.video_url} // Force reload on URL change
                controls
                controlsList="nodownload"
                preload="auto"
                className="w-full h-full object-contain"
                playsInline
                src={`${API_BASE_URL}/api/proxy-video?url=${encodeURIComponent(result.video_url)}`}
              >
                <p className="text-white p-4">
                  Your browser does not support the video tag.
                  <br />
                  <a 
                    href={result.video_url} 
                    className="text-blue-400 underline" 
                    download="interview_video.mp4"
                  >
                    Download video instead
                  </a>
                </p>
              </video>
            </div>
            <div className="mt-4 flex gap-2">
              <a
                href={result.video_url}
                download="interview_video.mp4"
                className="inline-flex items-center text-sm px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90"
              >
                📥 Download Video
              </a>
              <a
                href={result.video_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-sm px-4 py-2 border rounded-lg hover:bg-gray-50"
              >
                🔗 Open Direct Link →
              </a>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              💡 Video is streamed through API server for better browser compatibility
            </p>
          </div>

          {/* Full Report */}
          <div className="bg-card rounded-lg border p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold">Full Proctoring Report</h3>
              <Button
                onClick={() => {
                  const blob = new Blob([result.report_text || ''], { type: 'text/plain' });
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `proctoring_report_${Date.now()}.txt`;
                  document.body.appendChild(a);
                  a.click();
                  window.URL.revokeObjectURL(url);
                  document.body.removeChild(a);
                }}
                variant="outline"
                size="sm"
              >
                Download Report
              </Button>
            </div>
            <pre className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm font-mono whitespace-pre-wrap">
              {result.report_text}
            </pre>
          </div>
        </div>
      )}

      {/* Loading State */}
      {analyzing && (
        <div className="bg-card rounded-lg border p-12 text-center">
          <div className="flex flex-col items-center justify-center">
            <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
            <h3 className="text-xl font-semibold mb-2">Analyzing Video</h3>
            <p className="text-muted-foreground mb-4">
              This may take several minutes depending on video length...
            </p>
            <div className="text-sm text-muted-foreground space-y-1">
              <p>✓ Downloading video from Azure</p>
              <p>✓ Processing frames</p>
              <p>✓ Detecting faces and gaze direction</p>
              <p>✓ Generating report</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

