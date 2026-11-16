import { beforeEach, describe, expect, it } from 'vitest';
import { useStudioEditor } from './editorStore';

describe('studio editor board semantics', () => {
    beforeEach(() => {
        useStudioEditor.setState({
            boardAreas: {},
            boardLinks: [],
            boardMode: 'select',
            linkKind: 'thread',
            selectedAreaId: null,
        });
    });

    it('creates, updates and removes areas', () => {
        const id = useStudioEditor.getState().addArea({
            title: 'Act I',
            x: 0,
            y: 0,
            width: 640,
            height: 400,
            color: '#06b2eb',
        });
        expect(useStudioEditor.getState().boardAreas[id].title).toBe('Act I');
        expect(useStudioEditor.getState().selectedAreaId).toBe(id);

        useStudioEditor.getState().updateArea(id, { title: 'Act II', width: 720 });
        expect(useStudioEditor.getState().boardAreas[id]).toMatchObject({ title: 'Act II', width: 720 });

        useStudioEditor.getState().removeArea(id);
        expect(useStudioEditor.getState().boardAreas[id]).toBeUndefined();
        expect(useStudioEditor.getState().selectedAreaId).toBeNull();
    });

    it('dedupes typed links and rejects self links', () => {
        const state = useStudioEditor.getState();
        state.addLink('a', 'b', 'thread');
        state.addLink('b', 'a', 'thread');
        state.addLink('a', 'b', 'blocking');
        state.addLink('a', 'a', 'thread');

        const links = useStudioEditor.getState().boardLinks;
        expect(links).toHaveLength(2);
        expect(links.map((link) => link.kind).sort()).toEqual(['blocking', 'thread']);

        useStudioEditor.getState().removeLink(links[0].id);
        expect(useStudioEditor.getState().boardLinks).toHaveLength(1);
    });
});
