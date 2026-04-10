import React, { useState } from 'react';
import type { CardData } from '../types/game';
import Card from './Card';
import './TalonDrawer.css';

interface TalonDrawerProps {
  talonGroups: CardData[][] | null;
  putDown: CardData[];
  chosenGroup?: number | null;
  side?: 'left' | 'right';
}

export default function TalonDrawer({ talonGroups, putDown, chosenGroup, side = 'left' }: TalonDrawerProps) {
  const [open, setOpen] = useState(false);

  const hasContent = (talonGroups && talonGroups.length > 0) || putDown.length > 0;
  if (!hasContent) return null;

  return (
    <div className={`talon-drawer talon-drawer-${side} ${open ? 'talon-drawer-open' : ''}`}>
      <button
        className="talon-drawer-tab"
        onClick={() => setOpen(!open)}
        title={open ? 'Close talon info' : 'Show talon info'}
      >
        <span className="talon-drawer-tab-icon">📦</span>
        <span className="talon-drawer-tab-label">Talon</span>
      </button>

      <div className="talon-drawer-panel">
        <div className="talon-drawer-header">
          <h4>Talon &amp; Put Down</h4>
          <button className="talon-drawer-close" onClick={() => setOpen(false)}>×</button>
        </div>

        <div className="talon-drawer-body">
          {talonGroups && talonGroups.length > 0 && (
            <div className="talon-drawer-section">
              <h5>Talon Groups</h5>
              <div className="talon-drawer-groups">
                {talonGroups.map((group, i) => (
                  <div
                    key={i}
                    className={`talon-drawer-group ${chosenGroup === i ? 'talon-drawer-group-chosen' : ''}`}
                  >
                    {chosenGroup === i && <span className="talon-chosen-badge">Chosen</span>}
                    <div className="talon-drawer-cards">
                      {group.map((card, j) => (
                        <Card key={j} card={card} small />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {putDown.length > 0 && (
            <div className="talon-drawer-section">
              <h5>Put Down</h5>
              <div className="talon-drawer-cards">
                {putDown.map((card, j) => (
                  <Card key={j} card={card} small />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
