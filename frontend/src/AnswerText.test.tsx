// @vitest-environment jsdom
import {afterEach,it,expect} from 'vitest';
import {render,screen,cleanup} from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import AnswerText from './AnswerText';
afterEach(cleanup);
it('renders safe Markdown without raw markers',()=>{const {container}=render(<AnswerText text={'**Summary**\n- Use `POST /jobs`.\n- Inspect evidence.'}/>);expect(screen.getByText('Summary').tagName).toBe('STRONG');expect(screen.getByText('POST /jobs').tagName).toBe('CODE');expect(container.querySelectorAll('li')).toHaveLength(2);expect(container.textContent).not.toContain('**');});
it('never executes raw HTML',()=>{const {container}=render(<AnswerText text={'**Source**\n<script>alert(1)</script>'}/>);expect(container.querySelector('script')).toBeNull();expect(screen.getByText('<script>alert(1)</script>')).toBeInTheDocument();});
