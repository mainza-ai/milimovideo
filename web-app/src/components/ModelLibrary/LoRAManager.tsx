import React, { useEffect, useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useTimelineStore } from '../../stores/timelineStore';
import {
    Zap, ZapOff, Layers, Trash2, Plus
} from 'lucide-react';
import './LoRAManager.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface ActiveLoRA {
    model_id: string;
    name: string;
    strength: number;
    role: string;
}

export const LoRAManager: React.FC = () => {
    const {
        models,
        addToast,
    } = useTimelineStore(useShallow(state => ({
        models: state.models,
        addToast: state.addToast,
    })));

    const [activeLoras, setActiveLoras] = React.useState<ActiveLoRA[]>([]);
    const [slot, setSlot] = React.useState<'video' | 'image'>('video');
    const [isLoading, setIsLoading] = React.useState(false);

    // Available (downloaded) LoRAs
    const availableLoras = models.filter(m =>
        (m.type === 'ic_lora' || m.type === 'camera_lora') &&
        (m.status === 'downloaded' || m.status === 'active')
    );

    // Check if base model is active for this slot
    const baseActive = models.some(m =>
        m.type === 'base' &&
        m.status === 'active' &&
        ((slot === 'video' && (m.family === 'ltx2' || m.family === 'ltx23')) ||
         (slot === 'image' && m.family === 'flux2'))
    );

    // Fetch active LoRAs
    const fetchActiveLoras = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/models/loras/${slot}`);
            if (res.ok) {
                const data = await res.json();
                setActiveLoras(data.loras || []);
            }
        } catch (err) {
            console.error('[LoRAManager] Failed to fetch active LoRAs:', err);
        }
    }, [slot]);

    useEffect(() => {
        fetchActiveLoras();
    }, [fetchActiveLoras]);

    // Add LoRA
    const handleAddLora = async (modelId: string) => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE_URL}/models/${modelId}/lora`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strength: 1.0 }),
            });
            if (!res.ok) {
                const err = await res.json();
                addToast(err.detail || 'Failed to add LoRA', 'error');
                return;
            }
            const data = await res.json();
            setActiveLoras(data.loras || []);
            addToast(`LoRA added: ${data.message}`, 'success');
        } catch (err) {
            addToast('Failed to add LoRA', 'error');
        } finally {
            setIsLoading(false);
        }
    };

    // Remove LoRA
    const handleRemoveLora = async (modelId: string) => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE_URL}/models/${modelId}/lora`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                const err = await res.json();
                addToast(err.detail || 'Failed to remove LoRA', 'error');
                return;
            }
            const data = await res.json();
            setActiveLoras(data.loras || []);
            addToast('LoRA removed', 'info');
        } catch (err) {
            addToast('Failed to remove LoRA', 'error');
        } finally {
            setIsLoading(false);
        }
    };

    // Update strength
    const handleStrengthChange = async (modelId: string, strength: number) => {
        // Optimistic update
        setActiveLoras(prev =>
            prev.map(l => l.model_id === modelId ? { ...l, strength } : l)
        );

        try {
            const res = await fetch(`${API_BASE_URL}/models/${modelId}/lora`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strength }),
            });
            if (!res.ok) {
                // Revert on failure
                fetchActiveLoras();
                addToast('Failed to update strength', 'error');
            }
        } catch (err) {
            fetchActiveLoras();
        }
    };

    // Check if a LoRA is currently active
    const isActive = (modelId: string) =>
        activeLoras.some(l => l.model_id === modelId);

    // LoRAs not yet stacked
    const availableToAdd = availableLoras.filter(m => !isActive(m.id));

    return (
        <div className="lora-manager">
            <div className="lora-manager__header">
                <div className="lora-manager__title">
                    <Layers size={14} />
                    LoRA Stack
                </div>
                <div className="lora-manager__slot-toggle">
                    <button
                        className={`lora-slot-btn ${slot === 'video' ? 'lora-slot-btn--active' : ''}`}
                        onClick={() => setSlot('video')}
                    >
                        Video
                    </button>
                    <button
                        className={`lora-slot-btn ${slot === 'image' ? 'lora-slot-btn--active' : ''}`}
                        onClick={() => setSlot('image')}
                    >
                        Image
                    </button>
                </div>
            </div>

            {/* Base model status */}
            {!baseActive && (
                <div className="lora-manager__warning">
                    ⚠️ No base model active for {slot} pipeline. Activate a base model first.
                </div>
            )}

            {/* Active LoRA Stack */}
            {activeLoras.length > 0 && (
                <div className="lora-manager__section">
                    <div className="lora-manager__section-title">Active Stack</div>
                    <div className="lora-stack">
                        {activeLoras.map((lora) => (
                            <div key={lora.model_id} className="lora-item lora-item--active">
                                <div className="lora-item__info">
                                    <Zap size={10} className="lora-item__icon lora-item__icon--active" />
                                    <span className="lora-item__name">{lora.name}</span>
                                    <span className={`lora-item__role lora-item__role--${lora.role}`}>
                                        {lora.role.replace(/_/g, ' ')}
                                    </span>
                                </div>
                                <div className="lora-item__controls">
                                    <input
                                        type="range"
                                        className="lora-item__slider"
                                        min="0"
                                        max="2"
                                        step="0.05"
                                        value={lora.strength}
                                        onChange={(e) => handleStrengthChange(lora.model_id, parseFloat(e.target.value))}
                                        title={`Strength: ${lora.strength.toFixed(2)}`}
                                    />
                                    <span className="lora-item__strength">
                                        {lora.strength.toFixed(2)}
                                    </span>
                                    <button
                                        className="lora-item__remove"
                                        onClick={() => handleRemoveLora(lora.model_id)}
                                        disabled={isLoading}
                                        title="Remove LoRA"
                                    >
                                        <Trash2 size={10} />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Available LoRAs to Add */}
            {availableToAdd.length > 0 && baseActive && (
                <div className="lora-manager__section">
                    <div className="lora-manager__section-title">Available LoRAs</div>
                    <div className="lora-available">
                        {availableToAdd.map((model) => (
                            <div key={model.id} className="lora-item lora-item--available">
                                <div className="lora-item__info">
                                    <ZapOff size={10} className="lora-item__icon" />
                                    <span className="lora-item__name">{model.name}</span>
                                    <span className={`lora-item__role lora-item__role--${model.type}`}>
                                        {model.type.replace(/_/g, ' ')}
                                    </span>
                                </div>
                                <button
                                    className="lora-item__add"
                                    onClick={() => handleAddLora(model.id)}
                                    disabled={isLoading}
                                    title="Add to stack"
                                >
                                    <Plus size={10} /> Add
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {availableLoras.length === 0 && (
                <div className="lora-manager__empty">
                    No LoRAs downloaded. Go to the Model Library to download LoRAs.
                </div>
            )}
        </div>
    );
};
