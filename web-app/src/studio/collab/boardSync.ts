import * as Y from 'yjs';
import type { BoardPosition } from '../state/editorStore';

export interface BoardCollab {
    publish: (
        positions: Record<string, BoardPosition>,
        pinned: Record<string, boolean>,
    ) => void;
    stop: () => void;
}

/**
 * Same-machine, same-origin collaboration spike. Yjs owns convergence; a
 * BroadcastChannel carries updates between tabs. No server, no cloud.
 */
export function startBoardCollab(
    projectId: string,
    onRemote: (
        positions: Record<string, BoardPosition>,
        pinned: Record<string, boolean>,
    ) => void,
): BoardCollab {
    const doc = new Y.Doc();
    const yPositions = doc.getMap<BoardPosition>('positions');
    const yPinned = doc.getMap<boolean>('pinned');
    const channel = new BroadcastChannel(`milimo-board-${projectId}`);
    let remoteApplying = false;

    const snapshot = () => {
        remoteApplying = true;
        onRemote(
            Object.fromEntries(yPositions.entries()),
            Object.fromEntries(yPinned.entries()),
        );
    };

    doc.on('update', (update, origin) => {
        if (origin !== 'remote') channel.postMessage(update);
    });
    channel.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        Y.applyUpdate(doc, new Uint8Array(event.data), 'remote');
    };
    yPositions.observe((_event, transaction) => {
        if (transaction.origin === 'remote') snapshot();
    });
    yPinned.observe((_event, transaction) => {
        if (transaction.origin === 'remote') snapshot();
    });

    return {
        publish: (positions, pinned) => {
            if (remoteApplying) {
                remoteApplying = false;
                return;
            }
            doc.transact(() => {
                for (const [id, position] of Object.entries(positions)) {
                    yPositions.set(id, position);
                }
                for (const id of [...yPositions.keys()]) {
                    if (!(id in positions)) yPositions.delete(id);
                }
                for (const [id, value] of Object.entries(pinned)) {
                    yPinned.set(id, value);
                }
                for (const id of [...yPinned.keys()]) {
                    if (!(id in pinned)) yPinned.delete(id);
                }
            });
        },
        stop: () => {
            yPositions.unobserve(snapshot);
            yPinned.unobserve(snapshot);
            channel.close();
            doc.destroy();
        },
    };
}
