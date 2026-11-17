import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { startBoardCollab } from './boardSync';

class FakeChannel {
    static rooms = new Map<string, Set<FakeChannel>>();
    onmessage: ((event: { data: Uint8Array }) => void) | null = null;
    name: string;

    constructor(name: string) {
        this.name = name;
        const room = FakeChannel.rooms.get(name) || new Set<FakeChannel>();
        room.add(this);
        FakeChannel.rooms.set(name, room);
    }

    postMessage(data: Uint8Array) {
        for (const other of FakeChannel.rooms.get(this.name) || []) {
            if (other !== this) other.onmessage?.({ data: new Uint8Array(data) });
        }
    }

    close() {
        FakeChannel.rooms.get(this.name)?.delete(this);
    }
}

describe('board collaboration spike', () => {
    beforeEach(() => {
        vi.stubGlobal('BroadcastChannel', FakeChannel);
        FakeChannel.rooms.clear();
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('converges board positions between two docs', async () => {
        let left: Record<string, unknown> = {};
        let right: Record<string, unknown> = {};
        const a = startBoardCollab('room', (positions) => { left = positions; });
        const b = startBoardCollab('room', (positions) => { right = positions; });

        const positions = { shot1: { x: 120, y: 80, width: 300, height: 238 } };
        a.publish(positions, { shot1: true });
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(right).toEqual(positions);
        expect(Object.keys(left)).toHaveLength(0);

        a.stop();
        b.stop();
    });
});
