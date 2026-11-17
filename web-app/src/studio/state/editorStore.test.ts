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
            selectedShotId: null,
            selectedShotIds: [],
            boardPositions: {},
            boardPinned: {},
            boardPast: [],
            boardFuture: [],
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

    it('tracks a multi-selection with a primary shot', () => {
        const state = useStudioEditor.getState();
        state.selectShots(['a', 'b']);
        expect(useStudioEditor.getState().selectedShotIds).toEqual(['a', 'b']);
        expect(useStudioEditor.getState().selectedShotId).toBe('b');

        useStudioEditor.getState().toggleShotSelection('a');
        expect(useStudioEditor.getState().selectedShotIds).toEqual(['b']);
        expect(useStudioEditor.getState().selectedShotId).toBe('b');

        useStudioEditor.getState().toggleShotSelection('b');
        expect(useStudioEditor.getState().selectedShotIds).toEqual([]);
        expect(useStudioEditor.getState().selectedShotId).toBeNull();
    });

    it('undoes and redoes board layout changes', () => {
        const state = useStudioEditor.getState();
        state.setBoardPosition('undo-shot', { x: 10, y: 10, width: 300, height: 238 });
        useStudioEditor.getState().pushBoardHistory();
        useStudioEditor.getState().setBoardPosition('undo-shot', { x: 240, y: 80 });

        expect(useStudioEditor.getState().boardPositions['undo-shot'].x).toBe(240);
        useStudioEditor.getState().undoBoard();
        expect(useStudioEditor.getState().boardPositions['undo-shot']).toMatchObject({ x: 10, y: 10 });
        useStudioEditor.getState().redoBoard();
        expect(useStudioEditor.getState().boardPositions['undo-shot'].x).toBe(240);
    });

    it('saves, replaces and deletes recipes', () => {
        useStudioEditor.setState({ recipes: [] });
        useStudioEditor.getState().saveRecipe({ name: 'Rain pass', mode: 'direct', text: 'add rain' });
        useStudioEditor.getState().saveRecipe({ name: 'Rain pass', mode: 'direct', text: 'add heavy rain' });
        expect(useStudioEditor.getState().recipes).toHaveLength(1);
        expect(useStudioEditor.getState().recipes[0].text).toBe('add heavy rain');

        useStudioEditor.getState().deleteRecipe(useStudioEditor.getState().recipes[0].id);
        expect(useStudioEditor.getState().recipes).toEqual([]);
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
