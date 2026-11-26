import { useState } from 'react';
import { buildProgressView } from '../utils/progress';
import './ProgressCard.css';

export default function ProgressCard({
  events = [],
  outcome = null,
  elapsed = 0,
  fallbackTitle = null,
  fallbackContent = null,
}) {
  const isPending = outcome === null;
  const isFailed = outcome === 'failed';
  const [userExpanded, setUserExpanded] = useState(null);

  const view =
    events.length > 0 || isPending || isFailed
      ? buildProgressView(events, outcome, elapsed)
      : {
          title: fallbackTitle || 'Quá trình trả lời',
          status: 'done',
          steps: (fallbackContent || '')
            .split('\n')
            .filter(Boolean)
            .map((line) => ({
              icon: line.includes('✓') ? '✓' : '•',
              text: line.replace(/^-+\s*[✓◉]?\s*/, ''),
              complete: true,
            })),
        };

  const open = userExpanded !== null ? userExpanded : isPending || isFailed;

  return (
    <details
      className={`progress-card ${isPending ? 'pending' : ''} ${isFailed ? 'failed' : ''}`}
      open={open}
      onToggle={(event) => {
        setUserExpanded(event.currentTarget.open);
      }}
    >
      <summary className="progress-summary">
        {isPending ? (
          <span className="progress-spinner" aria-hidden="true" />
        ) : (
          <span className="progress-badge" aria-hidden="true">
            {isFailed ? '!' : '✓'}
          </span>
        )}
        <span className="progress-title">{view.title}</span>
      </summary>
      {view.steps.length > 0 && (
        <ul className="progress-steps">
          {view.steps.map((step, index) => (
            <li
              key={`${step.text}-${index}`}
              className={`progress-step ${step.complete ? 'complete' : 'active'}`}
            >
              <span className="step-icon" aria-hidden="true">
                {step.icon}
              </span>
              <span>{step.text}</span>
            </li>
          ))}
        </ul>
      )}
    </details>
  );
}
