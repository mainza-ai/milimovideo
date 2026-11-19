import { describe, expect, it } from 'vitest';
import { requestConfirm, useConfirmStore } from './confirmStore';

describe('confirm dialog store', () => {
    it('resolves true on confirm and false on cancel', async () => {
        const approved = requestConfirm({ title: 'Delete?', tone: 'danger' });
        expect(useConfirmStore.getState().current?.options.title).toBe('Delete?');
        useConfirmStore.getState().close(true);
        await expect(approved).resolves.toBe(true);

        const declined = requestConfirm({ title: 'Overwrite?' });
        useConfirmStore.getState().close(false);
        await expect(declined).resolves.toBe(false);
    });
});
