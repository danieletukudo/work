'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/livekit/button';

interface Evaluation {
  filename: string;
  candidate_id: string;
  candidate_name: string;
  job_id: string;
  overall_score: number;
  hiring_recommendation: string;
  created_at: string;
  file_size: number;
  video_url?: string;
  transcript_url?: string;
}

interface EvaluationStatus {
  status: 'no_evaluations' | 'ready' | 'in_progress';
  latest_evaluation: {
    filename: string;
    modified_time: number;
    modified_ago_seconds: number;
    file_size: number;
    is_recent: boolean;
  } | null;
  evaluation_count: number;
}

export function EvaluationDownload() {
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastEvaluationCount, setLastEvaluationCount] = useState(0);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const evaluationStartTimeRef = useRef<number | null>(null);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';

  useEffect(() => {
    fetchEvaluations();
    startPolling();
    
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  const fetchEvaluations = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/evaluations`);
      if (!response.ok) {
        throw new Error('Failed to fetch evaluations');
      }
      const data = await response.json();
      setEvaluations(data.evaluations || []);
      setLastEvaluationCount(data.evaluations?.length || 0);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load evaluations');
      console.error('Error fetching evaluations:', err);
    } finally {
      setLoading(false);
    }
  };

  const checkEvaluationStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluations/status`);
      if (!response.ok) {
        return;
      }
      const status: EvaluationStatus = await response.json();
      
      // If we have a new evaluation (count increased), refresh the list
      if (status.evaluation_count > lastEvaluationCount) {
        setLastEvaluationCount(status.evaluation_count);
        fetchEvaluations();
        setEvaluating(false);
        evaluationStartTimeRef.current = null;
        return;
      }
      
      // Check if evaluation is in progress
      if (status.latest_evaluation) {
        const { modified_ago_seconds, file_size } = status.latest_evaluation;
        
        // If file was modified very recently (< 2 minutes), evaluation might be in progress
        if (modified_ago_seconds < 120) {
          // Show evaluating state if:
          // 1. File is very recent (< 30 seconds) - likely just started
          // 2. Or we're already in evaluating state and file is still recent
          if (modified_ago_seconds < 30 || (evaluating && modified_ago_seconds < 120)) {
            if (!evaluating) {
              setEvaluating(true);
              evaluationStartTimeRef.current = Date.now();
            }
          }
        } else if (evaluating && modified_ago_seconds >= 120) {
          // Evaluation likely completed (file hasn't been modified in 2+ minutes)
          // But wait a bit more to ensure it's really done
          if (modified_ago_seconds > 180) {
            setEvaluating(false);
            evaluationStartTimeRef.current = null;
            fetchEvaluations(); // Refresh to get latest
          }
        }
      }
    } catch (err) {
      console.error('Error checking evaluation status:', err);
    }
  };

  const startPolling = () => {
    // Poll every 3 seconds for evaluation status
    pollingIntervalRef.current = setInterval(() => {
      checkEvaluationStatus();
    }, 3000);
  };

  const downloadEvaluation = async (filename: string, format: 'json' | 'txt') => {
    try {
      const endpoint = format === 'json' 
        ? `${API_BASE_URL}/api/evaluations/${filename}`
        : `${API_BASE_URL}/api/evaluations/${filename}/txt`;
      
      const response = await fetch(endpoint);
      if (!response.ok) {
        throw new Error(`Failed to download ${format.toUpperCase()} file`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = format === 'json' ? filename : filename.replace('.json', '.txt');
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error(`Error downloading ${format.toUpperCase()}:`, err);
      alert(`Failed to download ${format.toUpperCase()} file`);
    }
  };

  // Loading spinner component
  const LoadingSpinner = () => (
    <div className="flex items-center justify-center p-8">
      <div className="relative">
        <div className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full animate-spin"></div>
      </div>
      <div className="ml-4">
        <p className="text-lg font-medium">Evaluating Interview...</p>
        <p className="text-sm text-muted-foreground mt-1">
          This may take a few moments. Please wait.
        </p>
        {evaluationStartTimeRef.current && (
          <p className="text-xs text-muted-foreground mt-1">
            Started {Math.floor((Date.now() - evaluationStartTimeRef.current) / 1000)}s ago
          </p>
        )}
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="p-4 text-center">
        <div className="flex items-center justify-center">
          <div className="w-8 h-8 border-4 border-primary/20 border-t-primary rounded-full animate-spin mr-3"></div>
          <p className="text-muted-foreground">Loading evaluations...</p>
        </div>
      </div>
    );
  }

  // Show evaluating state prominently
  if (evaluating) {
    return (
      <div className="p-4">
        <div className="border-2 border-primary/30 rounded-lg bg-primary/5 p-6">
          <LoadingSpinner />
        </div>
        {evaluations.length > 0 && (
          <div className="mt-4">
            <h3 className="text-sm font-medium mb-2 text-muted-foreground">Previous Evaluations</h3>
            <div className="space-y-3">
              {evaluations.slice(0, 3).map((eval_item) => (
                <div
                  key={eval_item.filename}
                  className="border rounded-lg p-3 opacity-60"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-sm">{eval_item.candidate_name}</p>
                      <p className="text-xs text-muted-foreground">
                        Score: {eval_item.overall_score}/10
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-center">
        <p className="text-destructive">{error}</p>
        <Button onClick={fetchEvaluations} className="mt-2" variant="outline">
          Retry
        </Button>
      </div>
    );
  }

  if (evaluations.length === 0) {
    return (
      <div className="p-4 text-center">
        <p className="text-muted-foreground">No evaluations available yet.</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Interview Evaluations</h2>
        <Button onClick={fetchEvaluations} variant="outline" size="sm">
          Refresh
        </Button>
      </div>

      <div className="space-y-3">
        {evaluations.map((eval_item) => (
          <div
            key={eval_item.filename}
            className="border rounded-lg p-4 space-y-2"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h3 className="font-medium">{eval_item.candidate_name}</h3>
                <p className="text-sm text-muted-foreground">
                  Candidate ID: {eval_item.candidate_id} | Job ID: {eval_item.job_id}
                </p>
                <div className="flex items-center gap-4 mt-2 text-sm">
                  <span>
                    Score: <strong>{eval_item.overall_score}/10</strong>
                  </span>
                  <span>
                    Recommendation: <strong className="uppercase">{eval_item.hiring_recommendation}</strong>
                  </span>
                  <span className="text-muted-foreground">
                    {new Date(eval_item.created_at).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 pt-2">
              <Button
                onClick={() => downloadEvaluation(eval_item.filename, 'json')}
                variant="outline"
                size="sm"
              >
                📥 JSON
              </Button>
              <Button
                onClick={() => downloadEvaluation(eval_item.filename, 'txt')}
                variant="outline"
                size="sm"
              >
                📄 TXT
              </Button>
              {eval_item.video_url && (
                <a
                  href={`/proctoring?video_url=${encodeURIComponent(eval_item.video_url)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Button
                    variant="outline"
                    size="sm"
                  >
                    🎥 Analyze Video
                  </Button>
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

