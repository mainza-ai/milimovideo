import React from 'react';
import type { MatchedElement } from '../../stores/types';
import { getAssetUrl } from '../../config';

/**
 * Reusable component that displays matched element badges.
 * Used in both ScriptInput preview and StoryboardShotCard.
 * 
 * Features:
 * - Circular avatar with element image or type icon
 * - Color-coded by element type (purple=character, green=location, blue=object)
 * - Tooltip with element name, type, and confidence
 */

interface ElementBadgeProps {
    element: MatchedElement;
    size?: 'sm' | 'md';
}

const TYPE_COLORS: Record<string, string> = {
    character: 'rgba(168,85,247,0.6)',   // Purple
    location: 'rgba(34,197,94,0.6)',     // Green
    object: 'rgba(59,130,246,0.6)',      // Blue
};

const TYPE_ICONS: Record<string, string> = {
    character: '👤',
    location: '📍',
    object: '🔧',
};

export const ElementBadge: React.FC<ElementBadgeProps> = ({ element, size = 'sm' }) => {
    const color = TYPE_COLORS[element.element_type] || 'rgba(255,255,255,0.3)';
    const icon = TYPE_ICONS[element.element_type] || '❓';
    const dim = size === 'sm' ? 20 : 26;
    const fontSize = size === 'sm' ? '8px' : '10px';
    const imageUrl = element.image_url ? getAssetUrl(element.image_url) : null;
    const confidencePercent = Math.round(element.confidence * 100);

    return (
        <div
            title={`${element.element_name} (${element.element_type}) — ${confidencePercent}% confidence`}
            style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                padding: '2px 6px 2px 2px',
                borderRadius: 12,
                background: color,
                border: `1px solid ${color.replace('0.6', '0.8')}`,
                cursor: 'default',
                transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
                (e.currentTarget as HTMLDivElement).style.transform = 'scale(1.05)';
                (e.currentTarget as HTMLDivElement).style.boxShadow = `0 0 8px ${color}`;
            }}
            onMouseLeave={(e) => {
                (e.currentTarget as HTMLDivElement).style.transform = 'scale(1)';
                (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
            }}
        >
            {imageUrl ? (
                <img
                    src={imageUrl}
                    alt={element.element_name}
                    style={{
                        width: dim,
                        height: dim,
                        borderRadius: '50%',
                        objectFit: 'cover',
                        border: '1px solid rgba(255,255,255,0.3)',
                    }}
                />
            ) : (
                <span
                    style={{
                        width: dim,
                        height: dim,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: 'rgba(0,0,0,0.3)',
                        fontSize: size === 'sm' ? '10px' : '14px',
                    }}
                >
                    {icon}
                </span>
            )}
            <span style={{ fontSize, color: 'white', fontWeight: 500, whiteSpace: 'nowrap' }}>
                {element.element_name}
            </span>
        </div>
    );
};


interface ElementBadgeRowProps {
    elements: MatchedElement[];
    size?: 'sm' | 'md';
    maxDisplay?: number;
}

/**
 * Displays a row of element badges, with overflow indicator.
 */
export const ElementBadgeRow: React.FC<ElementBadgeRowProps> = ({
    elements,
    size = 'sm',
    maxDisplay = 4,
}) => {
    if (!elements || elements.length === 0) return null;

    const displayed = elements.slice(0, maxDisplay);
    const overflow = elements.length - maxDisplay;

    return (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, alignItems: 'center' }}>
            {displayed.map((el) => (
                <ElementBadge key={el.element_id} element={el} size={size} />
            ))}
            {overflow > 0 && (
                <span
                    style={{
                        fontSize: '9px',
                        color: 'rgba(255,255,255,0.4)',
                        padding: '2px 4px',
                    }}
                >
                    +{overflow} more
                </span>
            )}
        </div>
    );
};
