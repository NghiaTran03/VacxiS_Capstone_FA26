import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import './CitationChip.css';

export default function CitationChip({ citation }) {
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState({ left: 12, top: 12 });
  const chipRef = useRef(null);
  const popupRef = useRef(null);
  const closeTimerRef = useRef(null);
  const tooltipId = useId();

  const clearTimer = useCallback(() => {
    if (closeTimerRef.current !== null) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);

  const closeNow = useCallback(() => {
    clearTimer();
    setOpen(false);
  }, [clearTimer]);

  const scheduleClose = useCallback(() => {
    clearTimer();
    closeTimerRef.current = setTimeout(() => {
      setOpen(false);
    }, 160);
  }, [clearTimer]);

  const updatePosition = useCallback(() => {
    const chip = chipRef.current;
    const popup = popupRef.current;
    if (!chip || !popup) return;
    const anchor = chip.getBoundingClientRect();
    if (anchor.bottom < 0 || anchor.top > window.innerHeight) {
      setOpen(false);
      return;
    }
    const box = popup.getBoundingClientRect();
    const left = Math.max(
      12,
      Math.min(anchor.left, window.innerWidth - box.width - 12),
    );
    const top =
      anchor.top > box.height + 20
        ? anchor.top - box.height - 10
        : Math.min(anchor.bottom + 10, window.innerHeight - box.height - 12);
    setCoords({ left, top: Math.max(12, top) });
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    const frame = requestAnimationFrame(updatePosition);
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        closeNow();
      }
    };
    const onScroll = (event) => {
      if (popupRef.current && !popupRef.current.contains(event.target)) {
        updatePosition();
      }
    };
    const onResize = () => closeNow();

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onResize);
    };
  }, [open, updatePosition, closeNow]);

  useEffect(() => () => clearTimer(), [clearTimer]);

  const openTooltip = () => {
    clearTimer();
    setOpen(true);
  };

  const handleBlur = (event) => {
    const nextTarget = event.relatedTarget;
    if (
      nextTarget &&
      (popupRef.current?.contains(nextTarget) || chipRef.current?.contains(nextTarget))
    ) {
      return;
    }
    scheduleClose();
  };

  const sharedProps = {
    ref: chipRef,
    className: 'vacxis-citation',
    'aria-label': `Nguồn ${citation.number}: ${citation.title}`,
    'aria-describedby': open ? tooltipId : undefined,
    onPointerEnter: openTooltip,
    onPointerLeave: scheduleClose,
    onFocus: openTooltip,
    onBlur: handleBlur,
  };

  const popup = open
    ? createPortal(
        <div
          ref={popupRef}
          id={tooltipId}
          role="tooltip"
          className="vacxis-source-preview"
          style={{ left: `${coords.left}px`, top: `${coords.top}px` }}
          onPointerEnter={clearTimer}
          onPointerLeave={scheduleClose}
          onBlur={handleBlur}
        >
          <strong className="source-title">{citation.title}</strong>
          <div className="source-heading">{citation.header}</div>
          <div className="source-excerpt">{citation.excerpt}</div>
          {citation.href ? (
            <a
              className="source-footer"
              href={citation.href}
              target="_blank"
              rel="noopener noreferrer"
            >
              Mở tài liệu ↗
            </a>
          ) : (
            <span className="source-footer">Ngữ cảnh đồ thị tri thức</span>
          )}
        </div>,
        document.body,
      )
    : null;

  return (
    <>
      {citation.href ? (
        <a
          {...sharedProps}
          href={citation.href}
          target="_blank"
          rel="noopener noreferrer"
        >
          {citation.number}
        </a>
      ) : (
        <span {...sharedProps} tabIndex={0}>
          {citation.number}
        </span>
      )}
      {popup}
    </>
  );
}
